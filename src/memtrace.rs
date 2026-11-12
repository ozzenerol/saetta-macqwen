//! `SAETTA_MEM_TRACE=1`: the server logs the process memory at points of a
//! request (start, prefill, snapshot saves, decode steps, end), to find what
//! takes the 1.2 GB that comes and goes with each request. Each line has the
//! phys_footprint (what Activity Monitor and `footprint` show), its change
//! since the last line, the lifetime peak, and the parts the kernel tags:
//! graphics (IOAccelerator memory charged to the process), purgeable
//! nonvolatile and volatile, and internal (anonymous) memory.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, Ordering};

/// Memory counters of this process, in bytes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counters {
    pub footprint: i64,
    pub peak: i64,
    pub internal: i64,
    pub graphics: i64,
    pub graphics_nofootprint: i64,
    pub purgeable_nonvolatile: i64,
    pub purgeable_volatile: i64,
}

#[cfg(target_os = "macos")]
pub fn counters() -> Option<Counters> {
    // task_vm_info from <mach/task_info.h>, read as 8-byte words: index 6
    // internal, 18 phys_footprint, 21 ledger_phys_footprint_peak, 22
    // ledger_purgeable_nonvolatile, 24 ledger_purgeable_volatile, 34
    // ledger_tag_graphics_footprint, 36 ledger_tag_graphics_nofootprint.
    unsafe extern "C" {
        static mach_task_self_: u32;
        fn task_info(task: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
    }
    const TASK_VM_INFO: i32 = 22;
    let mut words = [0i64; 64];
    let mut count = (std::mem::size_of_val(&words) / 4) as u32;
    let r = unsafe {
        task_info(
            mach_task_self_,
            TASK_VM_INFO,
            words.as_mut_ptr().cast(),
            &mut count,
        )
    };
    if r != 0 {
        return None;
    }
    // The kernel says how many 4-byte words it filled.
    let filled = count as usize / 2;
    let get = |i: usize| if i < filled { words[i] } else { -1 };
    Some(Counters {
        footprint: get(18),
        peak: get(21),
        internal: get(6),
        graphics: get(34),
        graphics_nofootprint: get(36),
        purgeable_nonvolatile: get(22),
        purgeable_volatile: get(24),
    })
}

#[cfg(not(target_os = "macos"))]
pub fn counters() -> Option<Counters> {
    None
}

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SAETTA_MEM_TRACE").is_ok_and(|v| v == "1"))
}

/// The footprint at the last mark, for the change.
static LAST: AtomicI64 = AtomicI64::new(0);

fn mb(bytes: i64) -> String {
    if bytes < 0 {
        "?".into()
    } else {
        format!("{:.0}", bytes as f64 / (1 << 20) as f64)
    }
}

/// Logs the counters with `label`, when SAETTA_MEM_TRACE=1.
pub fn mark(label: &str) {
    if !enabled() {
        return;
    }
    let Some(c) = counters() else {
        eprintln!("mem {label}: no counters on this OS");
        return;
    };
    let last = LAST.swap(c.footprint, Ordering::Relaxed);
    eprintln!(
        "mem {label}: footprint {} MB ({:+.0}), peak {}, graphics {} (+{} not in footprint), purgeable {} nonvolatile {} volatile, internal {}",
        mb(c.footprint),
        (c.footprint - last) as f64 / (1 << 20) as f64,
        mb(c.peak),
        mb(c.graphics),
        mb(c.graphics_nofootprint),
        mb(c.purgeable_nonvolatile),
        mb(c.purgeable_volatile),
        mb(c.internal),
    );
}

/// Marks "<label> start" now and "<label> end" when dropped.
pub struct Span(&'static str);

impl Span {
    pub fn new(label: &'static str) -> Span {
        mark(&format!("{label} start"));
        Span(label)
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        mark(&format!("{} end", self.0));
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn mark_without_the_variable_is_silent() {
        // Must not panic on any OS.
        super::mark("test");
    }
}
