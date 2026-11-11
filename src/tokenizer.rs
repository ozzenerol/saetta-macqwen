//! Byte-level BPE from a Hugging Face `tokenizer.json`.
//!
//! Encoding goes the way the HF tokenizer does it for Qwen:
//! 1. cut out added tokens such as `<|im_start|>` where they appear in the text;
//! 2. NFC-normalize the rest;
//! 3. split it with the pre-tokenizer regex from the file;
//! 4. run BPE merges on the UTF-8 bytes of each piece.
//!
//! The vocab in the file stores bytes as printable characters (the GPT-2
//! byte-to-unicode table). We turn them back into bytes at load time, so
//! BPE and decode work on plain bytes.

use fancy_regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

pub struct Tokenizer {
    /// Token bytes to id, for the BPE vocab.
    vocab: HashMap<Vec<u8>, u32>,
    /// Pair of ids to (merge rank, merged id).
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Bytes of every token, indexed by id. Empty for unused ids.
    bytes: Vec<Vec<u8>>,
    /// Added tokens with their ids, longest first so the longest match wins.
    added: Vec<(String, u32)>,
    special: Vec<bool>,
    nfc: bool,
    split: Regex,
}

impl Tokenizer {
    pub fn load(model_dir: &Path) -> Result<Tokenizer, Box<dyn Error>> {
        let path = model_dir.join("tokenizer.json");
        let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Tokenizer::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Tokenizer, Box<dyn Error>> {
        let root: Value = serde_json::from_str(text)?;
        let nfc = match root["normalizer"]["type"].as_str() {
            None => false,
            Some("NFC") => true,
            Some(other) => return Err(format!("normalizer {other} is not supported").into()),
        };
        let split = Regex::new(split_pattern(&root["pre_tokenizer"])?)?;

        let model = &root["model"];
        if model["type"] != "BPE"
            || model["byte_fallback"] == true
            || model["ignore_merges"] == true
        {
            return Err("only byte-level BPE without byte fallback is supported".into());
        }

        let to_byte = unicode_to_byte();
        let token_bytes = |token: &str| -> Result<Vec<u8>, String> {
            token
                .chars()
                .map(|c| to_byte.get(&c).copied())
                .collect::<Option<Vec<u8>>>()
                .ok_or_else(|| format!("token {token:?} is not byte-level"))
        };

        let mut vocab = HashMap::new();
        let mut bytes = Vec::new();
        for (token, id) in model["vocab"].as_object().ok_or("model has no vocab")? {
            let id = id.as_u64().ok_or("bad vocab id")? as u32;
            let b = token_bytes(token)?;
            set(&mut bytes, id, b.clone());
            vocab.insert(b, id);
        }

        let mut merges = HashMap::new();
        for (rank, merge) in model["merges"]
            .as_array()
            .ok_or("model has no merges")?
            .iter()
            .enumerate()
        {
            // Newer files store a merge as ["a", "b"], older ones as "a b".
            let (a, b) = match merge {
                Value::Array(pair) if pair.len() == 2 => (
                    pair[0].as_str().ok_or("bad merge")?,
                    pair[1].as_str().ok_or("bad merge")?,
                ),
                Value::String(s) => s.split_once(' ').ok_or("bad merge")?,
                _ => return Err("bad merge".into()),
            };
            let (a, b) = (token_bytes(a)?, token_bytes(b)?);
            let joined = [a.as_slice(), b.as_slice()].concat();
            let id = |t: &[u8]| {
                vocab
                    .get(t)
                    .copied()
                    .ok_or("merge uses a token not in the vocab")
            };
            merges.insert((id(&a)?, id(&b)?), (rank as u32, id(&joined)?));
        }

        let mut added = Vec::new();
        let mut special = vec![false; bytes.len()];
        for token in root["added_tokens"].as_array().ok_or("no added_tokens")? {
            let content = token["content"]
                .as_str()
                .ok_or("bad added token")?
                .to_string();
            let id = token["id"].as_u64().ok_or("bad added token id")? as u32;
            set(&mut bytes, id, content.as_bytes().to_vec());
            set(&mut special, id, token["special"] == true);
            added.push((content, id));
        }
        added.sort_by_key(|(content, _)| std::cmp::Reverse(content.len()));
        special.resize(bytes.len(), false);

        Ok(Tokenizer {
            vocab,
            merges,
            bytes,
            added,
            special,
            nfc,
            split,
        })
    }

    /// Ids run from 0 to `vocab_size() - 1`, with gaps where no token is defined.
    pub fn vocab_size(&self) -> usize {
        self.bytes.len()
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.added
            .iter()
            .find(|(content, _)| content == token)
            .map(|&(_, id)| id)
            .or_else(|| self.vocab.get(token.as_bytes()).copied())
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(false)
    }

    /// The raw bytes of one token. A token can hold part of a UTF-8 character.
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).map_or(&[], Vec::as_slice)
    }

    /// Encodes text. Added tokens written in the text, such as `<|im_end|>`,
    /// become their ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let (plain, token) = self.next_added(rest);
            self.encode_plain(&rest[..plain], &mut ids);
            match token {
                Some((len, id)) => {
                    ids.push(id);
                    rest = &rest[plain + len..];
                }
                None => break,
            }
        }
        ids
    }

    /// Joins the token bytes. Invalid UTF-8 becomes U+FFFD, as in HF decode.
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&id| self.token_bytes(id))
            .copied()
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Finds the first added token in `text`: the byte position where it
    /// starts and its (length, id). Without one, the position is the end.
    fn next_added(&self, text: &str) -> (usize, Option<(usize, u32)>) {
        for (pos, _) in text.char_indices() {
            let tail = &text[pos..];
            if let Some((content, id)) = self
                .added
                .iter()
                .find(|(c, _)| tail.starts_with(c.as_str()))
            {
                return (pos, Some((content.len(), *id)));
            }
        }
        (text.len(), None)
    }

    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let normalized: String;
        let text = if self.nfc {
            normalized = text.nfc().collect();
            &normalized
        } else {
            text
        };
        // Text between regex matches is kept as its own piece.
        let mut end = 0;
        for m in self.split.find_iter(text) {
            let m = m.expect("pre-tokenizer regex failed");
            if m.start() > end {
                self.bpe(&text.as_bytes()[end..m.start()], ids);
            }
            self.bpe(m.as_str().as_bytes(), ids);
            end = m.end();
        }
        if end < text.len() {
            self.bpe(&text.as_bytes()[end..], ids);
        }
    }

    /// Starts from single bytes and merges the pair with the lowest rank,
    /// leftmost first, until no pair has a merge.
    fn bpe(&self, piece: &[u8], ids: &mut Vec<u32>) {
        if let Some(&id) = self.vocab.get(piece) {
            ids.push(id);
            return;
        }
        let mut parts: Vec<u32> = piece
            .iter()
            .map(|b| self.vocab[std::slice::from_ref(b)])
            .collect();
        loop {
            let best = parts
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    self.merges
                        .get(&(w[0], w[1]))
                        .map(|&(rank, id)| (rank, i, id))
                })
                .min();
            let Some((_, i, id)) = best else { break };
            parts[i] = id;
            parts.remove(i + 1);
        }
        ids.extend(parts);
    }
}

/// The regex of a pre-tokenizer of the form
/// `Sequence[Split(regex, Isolated), ByteLevel(use_regex: false)]`.
fn split_pattern(pre: &Value) -> Result<&str, String> {
    let unsupported = || format!("pre_tokenizer is not supported: {pre}");
    if pre["type"] != "Sequence" {
        return Err(unsupported());
    }
    let steps = pre["pretokenizers"].as_array().ok_or_else(unsupported)?;
    let [split, byte_level] = steps.as_slice() else {
        return Err(unsupported());
    };
    let split_ok =
        split["type"] == "Split" && split["behavior"] == "Isolated" && split["invert"] == false;
    let byte_level_ok = byte_level["type"] == "ByteLevel"
        && byte_level["use_regex"] == false
        && byte_level["add_prefix_space"] == false;
    if !split_ok || !byte_level_ok {
        return Err(unsupported());
    }
    split["pattern"]["Regex"].as_str().ok_or_else(unsupported)
}

/// The GPT-2 table, reversed: printable bytes stand for themselves and the
/// other 68 bytes get the characters from U+0100 on, in byte order.
fn unicode_to_byte() -> HashMap<char, u8> {
    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff);
    let mut next = 0x100;
    let mut map = HashMap::new();
    for b in 0..=255u8 {
        let c = if printable(b) {
            char::from(b)
        } else {
            next += 1;
            char::from_u32(next - 1).unwrap()
        };
        map.insert(c, b);
    }
    map
}

fn set<T: Default + Clone>(v: &mut Vec<T>, id: u32, value: T) {
    let i = id as usize;
    if v.len() <= i {
        v.resize(i + 1, T::default());
    }
    v[i] = value;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny tokenizer: bytes a, b, c, space and newline, with the merges
    /// "a b" -> "ab", "ab c" -> "abc" and "Ġ a" -> " a".
    const SMALL: &str = r#"{
        "added_tokens": [{"id": 9, "content": "<|end|>", "special": true}],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": " ?\\p{L}+|\\s+"}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]},
        "model": {"type": "BPE", "byte_fallback": false, "ignore_merges": false,
            "vocab": {"a": 0, "b": 1, "c": 2, "Ġ": 3, "Ċ": 4, "ab": 5, "abc": 6, "Ġa": 7},
            "merges": [["a", "b"], "ab c", ["Ġ", "a"]]}
    }"#;

    #[test]
    fn byte_table_matches_gpt2() {
        let map = unicode_to_byte();
        assert_eq!(map.len(), 256);
        assert_eq!(map[&'Ġ'], b' ');
        assert_eq!(map[&'Ċ'], b'\n');
        assert_eq!(map[&'A'], b'A');
        assert_eq!(map[&'\u{100}'], 0);
    }

    #[test]
    fn merges_by_rank() {
        let t = Tokenizer::parse(SMALL).unwrap();
        assert_eq!(t.encode("abc"), vec![6]);
        assert_eq!(t.encode("abab a"), vec![5, 5, 7]);
        assert_eq!(t.encode("cab\n"), vec![2, 5, 4]);
        assert_eq!(t.encode("ab<|end|>c"), vec![5, 9, 2]);
        assert_eq!(t.decode(&[5, 9, 7]), "ab<|end|> a");
        assert!(t.is_special(9));
        assert!(!t.is_special(5));
        assert_eq!(t.token_id("<|end|>"), Some(9));
        assert_eq!(t.vocab_size(), 10);
    }
}
