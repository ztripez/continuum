use continuum_cdsl::ast::KernelRegistry;

fn main() {
    let registry = KernelRegistry::global();
    println!("Total kernels: {}", registry.ids().count());
    
    // Check dt namespace
    let dt_kernels: Vec<_> = registry.ids()
        .filter(|id| id.namespace == "dt")
        .collect();
    println!("dt namespace kernels: {}", dt_kernels.len());
    for id in &dt_kernels {
        println!("  - dt.{}", id.name);
    }
    
    // Check specifically for dt.decay
    if let Some(sig) = registry.get_by_name("dt", "decay") {
        println!("✅ Found dt.decay");
    } else {
        println!("❌ dt.decay NOT FOUND!");
    }
}
