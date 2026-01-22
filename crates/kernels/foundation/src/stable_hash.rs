//! Stable hashing utilities for deterministic IDs.
//!
//! Continuum requires that any derived identifier or pseudo-random stream be a
//! deterministic consequence of explicit inputs. These helpers provide a stable
//! FNV-1a 64-bit implementation for use across crates.
//!
//! NOTE: FNV-1a is **not** cryptographically secure.
//! It is used strictly for stable identifiers and deterministic derivations.

/// 64-bit FNV-1a offset basis.
pub const FNV1A_OFFSET_BASIS_64: u64 = 0xcbf29ce484222325;
/// 64-bit FNV-1a prime.
pub const FNV1A_PRIME_64: u64 = 0x0000_0100_0000_01B3;

/// Mix bytes into an existing FNV-1a 64-bit hash state.
///
/// This function implements the core FNV-1a update step: for each byte,
/// XOR it into the hash and multiply by the FNV prime.
///
/// # Parameters
/// - `hash`: Current hash state (use [`FNV1A_OFFSET_BASIS_64`] for fresh hash)
/// - `bytes`: Byte slice to mix into hash
///
/// # Returns
/// Updated hash state after processing all bytes.
///
/// # Example
/// ```
/// use continuum_foundation::stable_hash::{fnv1a64_mix, FNV1A_OFFSET_BASIS_64};
///
/// let hash = FNV1A_OFFSET_BASIS_64;
/// let hash = fnv1a64_mix(hash, b"hello");
/// let hash = fnv1a64_mix(hash, b"world");
/// ```
#[inline]
pub const fn fnv1a64_mix(mut hash: u64, bytes: &[u8]) -> u64 {
    let mut i = 0usize;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(FNV1A_PRIME_64);
        i += 1;
    }
    hash
}

/// Hash an arbitrary byte slice with FNV-1a 64-bit.
#[inline]
pub const fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_mix(FNV1A_OFFSET_BASIS_64, bytes)
}

/// Hash a UTF-8 string with FNV-1a 64-bit.
#[inline]
pub const fn fnv1a64_str(s: &str) -> u64 {
    fnv1a64(s.as_bytes())
}

/// Hash a dotted/namespace path expressed as parts, inserting `.` between parts.
///
/// This avoids allocations and makes path semantics explicit.
///
/// Example:
/// ```
/// # use continuum_foundation::stable_hash::fnv1a64_path;
/// const ID: u64 = fnv1a64_path(&["terra", "sim", "plates"]);
/// ```
#[inline]
pub const fn fnv1a64_path(parts: &[&str]) -> u64 {
    let mut h = FNV1A_OFFSET_BASIS_64;

    let mut i = 0usize;
    while i < parts.len() {
        h = fnv1a64_mix(h, parts[i].as_bytes());
        if i + 1 < parts.len() {
            h = fnv1a64_mix(h, b".");
        }
        i += 1;
    }

    h
}

/// Compile-time stable ID from a *string literal* stable path.
///
/// Usage:
/// ```
/// use continuum_foundation::stable_id;
/// const ID: u64 = stable_id!("terra.sim.plates");
/// ```
///
/// Intentionally only accepts string literals to keep it const + stable.
#[macro_export]
macro_rules! stable_id {
    ($path:literal) => {{ $crate::stable_hash::fnv1a64_str($path) }};
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a 64-bit reference test values.
    /// These are canonical test vectors from the FNV spec.
    /// See: http://www.isthe.com/chongo/tech/comp/fnv/
    #[test]
    fn fnv1a64_reference_values() {
        // Empty string
        assert_eq!(fnv1a64(b""), FNV1A_OFFSET_BASIS_64);

        // Single byte tests - verify the algorithm step by step
        // FNV-1a: hash = (hash XOR byte) * prime
        let a_hash = fnv1a64(b"a");
        let expected_a = (FNV1A_OFFSET_BASIS_64 ^ 0x61).wrapping_mul(FNV1A_PRIME_64);
        assert_eq!(a_hash, expected_a);

        // Multi-byte - verify incremental hashing
        let ab_hash = fnv1a64(b"ab");
        let expected_ab = ((FNV1A_OFFSET_BASIS_64 ^ 0x61).wrapping_mul(FNV1A_PRIME_64) ^ 0x62)
            .wrapping_mul(FNV1A_PRIME_64);
        assert_eq!(ab_hash, expected_ab);
    }

    /// Verify specific known hash values for regression testing.
    /// If these fail, the hash algorithm has changed (breaking determinism).
    #[test]
    fn fnv1a64_regression_values() {
        // These are fixed regression values computed from FNV-1a 64-bit
        // Do not change - any change indicates a breaking determinism change
        assert_eq!(fnv1a64(b"hello"), 11831194018420276491);
        assert_eq!(fnv1a64(b"hello world"), 8618312879776256743);

        // Stability: same input always produces same output
        let terra_hash = fnv1a64(b"terra");
        let plates_hash = fnv1a64(b"terra.plates");
        assert_eq!(fnv1a64(b"terra"), terra_hash);
        assert_eq!(fnv1a64(b"terra.plates"), plates_hash);

        // Uniqueness: different inputs produce different outputs
        assert_ne!(terra_hash, plates_hash);
    }

    /// Verify fnv1a64_str matches fnv1a64 for string bytes.
    #[test]
    fn fnv1a64_str_matches_bytes() {
        let strings = ["", "a", "hello", "terra.plates.velocity"];
        for s in strings {
            assert_eq!(fnv1a64_str(s), fnv1a64(s.as_bytes()));
        }
    }

    /// Test fnv1a64_mix incremental hashing.
    #[test]
    fn fnv1a64_mix_incremental() {
        // Mixing in chunks should equal mixing all at once
        let full = fnv1a64(b"helloworld");

        let mut incremental = FNV1A_OFFSET_BASIS_64;
        incremental = fnv1a64_mix(incremental, b"hello");
        incremental = fnv1a64_mix(incremental, b"world");

        assert_eq!(full, incremental);
    }

    /// Test fnv1a64_path joins parts with dots.
    #[test]
    fn fnv1a64_path_joins_with_dots() {
        // Path with parts should equal string with dots
        assert_eq!(fnv1a64_path(&["terra"]), fnv1a64_str("terra"));
        assert_eq!(
            fnv1a64_path(&["terra", "plates"]),
            fnv1a64_str("terra.plates")
        );
        assert_eq!(
            fnv1a64_path(&["terra", "plates", "velocity"]),
            fnv1a64_str("terra.plates.velocity")
        );
        assert_eq!(fnv1a64_path(&["a", "b", "c", "d"]), fnv1a64_str("a.b.c.d"));
    }

    /// Test fnv1a64_path with empty array and single element.
    #[test]
    fn fnv1a64_path_edge_cases() {
        // Empty array - should equal empty string
        assert_eq!(fnv1a64_path(&[]), fnv1a64_str(""));

        // Single element - no dots
        assert_eq!(fnv1a64_path(&["single"]), fnv1a64_str("single"));

        // Empty strings in array
        assert_eq!(fnv1a64_path(&["", ""]), fnv1a64_str("."));
        assert_eq!(fnv1a64_path(&["a", "", "b"]), fnv1a64_str("a..b"));
    }

    /// Verify the stable_id! macro produces correct values.
    #[test]
    fn stable_id_macro() {
        const ID1: u64 = stable_id!("terra.sim.plates");
        const ID2: u64 = stable_id!("terra.sim.plates");
        const ID3: u64 = stable_id!("terra.sim.velocity");

        // Same path, same ID
        assert_eq!(ID1, ID2);

        // Different paths, different IDs
        assert_ne!(ID1, ID3);

        // Matches runtime computation
        assert_eq!(ID1, fnv1a64_str("terra.sim.plates"));
    }

    /// Test hash stability - same input always produces same output.
    #[test]
    fn hash_is_stable() {
        let inputs = [
            b"".as_slice(),
            b"a",
            b"hello",
            b"terra.plates.velocity.x",
            b"\x00\x01\x02\xff", // binary data
        ];

        for input in inputs {
            let hash1 = fnv1a64(input);
            let hash2 = fnv1a64(input);
            let hash3 = fnv1a64(input);
            assert_eq!(hash1, hash2);
            assert_eq!(hash2, hash3);
        }
    }

    /// Test that different inputs produce different hashes (collision resistance).
    #[test]
    fn different_inputs_different_hashes() {
        let inputs = ["a", "b", "aa", "ab", "ba", "aaa", "terra", "terra."];

        for (i, a) in inputs.iter().enumerate() {
            for (j, b) in inputs.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        fnv1a64_str(a),
                        fnv1a64_str(b),
                        "Collision between '{}' and '{}'",
                        a,
                        b
                    );
                }
            }
        }
    }

    /// Test const evaluation - these should all compile.
    #[test]
    fn const_evaluation() {
        // All these are evaluated at compile time
        const OFFSET: u64 = FNV1A_OFFSET_BASIS_64;
        const PRIME: u64 = FNV1A_PRIME_64;
        const HASH: u64 = fnv1a64(b"test");
        const HASH_STR: u64 = fnv1a64_str("test");
        const PATH_HASH: u64 = fnv1a64_path(&["a", "b", "c"]);
        const MIX: u64 = fnv1a64_mix(OFFSET, b"data");

        // Verify they have expected properties
        assert_ne!(OFFSET, 0);
        assert_ne!(PRIME, 0);
        assert_eq!(HASH, HASH_STR);
        assert_eq!(PATH_HASH, fnv1a64_str("a.b.c"));
        assert_ne!(MIX, OFFSET);
    }
}
