// What changes from one token to the next. The CPU writes it before each
// command buffer; kernels that need it take it as one more buffer.
// Must match plan::TokenParams.
struct TokenParams {
    uint token;     // token id being processed
    uint position;  // its position in the sequence, from 0
    uint kv_len;    // entries in the KV cache after this token: position + 1
    uint pad;
};
