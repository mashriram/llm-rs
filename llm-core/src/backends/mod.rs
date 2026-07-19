pub mod candle;
pub mod vision;
pub mod audio;

// Internal sub-modules — not exposed through `pub use`; imported by candle.rs.
pub(crate) mod attention;
pub(crate) mod multimodal;
pub(crate) mod weights;

use parking_lot::Mutex;

/// Global active image path for the current inference request.
/// Set by the CLI/server before calling `forward_pass`.
pub static ACTIVE_IMAGE_PATH: Mutex<Option<String>> = parking_lot::const_mutex(None);

/// Global active audio path for the current inference request.
pub static ACTIVE_AUDIO_PATH: Mutex<Option<String>> = parking_lot::const_mutex(None);
