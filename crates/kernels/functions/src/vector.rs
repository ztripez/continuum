//! Vector Constructors
//!
//! Functions for constructing vector types.
//! Note: These return f64 for now as the DSL only supports f64.
//! In the future, these will return proper vector types.

use continuum_kernel_macros::kernel_fn;

/// Construct a 2D vector: `vec2(x, y)`
/// Returns the magnitude for now (until proper vector types are supported)
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn vec2(x: f64, y: f64) -> f64 {
    (x * x + y * y).sqrt()
}

/// Construct a 3D vector: `vec3(x, y, z)`
/// Returns the magnitude for now (until proper vector types are supported)
#[kernel_fn(namespace = "vector", category = "vector")]
pub fn vec3(x: f64, y: f64, z: f64) -> f64 {
    (x * x + y * y + z * z).sqrt()
}

/// Vector length/magnitude: `length(x, y, z)`
/// Variadic - works with 2 or 3 components
#[kernel_fn(namespace = "vector", category = "vector", variadic)]
pub fn length(args: &[f64]) -> f64 {
    args.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use continuum_kernel_registry::{Arity, get_in_namespace, is_known_in};

    #[test]
    fn test_vec2_registered() {
        assert!(is_known_in("vector", "vec2"));
        let desc = get_in_namespace("vector", "vec2").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(2));
        assert!(!desc.requires_dt());
    }

    #[test]
    fn test_vec3_registered() {
        assert!(is_known_in("vector", "vec3"));
        let desc = get_in_namespace("vector", "vec3").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(3));
        assert!(!desc.requires_dt());
    }
}
