//! GPU context management for wgpu device and queue.
//!
//! The [`GpuContext`] provides the core wgpu resources needed for compute
//! operations. It handles adapter selection, device creation, and provides
//! a stable interface for the rest of the GPU subsystem.

use crate::error::GpuError;

/// GPU context holding wgpu device and queue.
///
/// This is the foundation for all GPU operations. Create one context
/// and share it across all pipelines.
pub struct GpuContext {
    /// wgpu device for resource creation.
    pub(crate) device: wgpu::Device,
    /// wgpu queue for command submission.
    pub(crate) queue: wgpu::Queue,
    /// Adapter info for diagnostics.
    adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Create a new GPU context with default settings.
    ///
    /// This will select the best available GPU adapter and request a device
    /// with compute capabilities.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::NoAdapter`] if no compatible GPU is found.
    /// Returns [`GpuError::DeviceRequest`] if device creation fails.
    pub fn new() -> Result<Self, GpuError> {
        Self::with_options(GpuContextOptions::default())
    }

    /// Create a new GPU context with custom options.
    pub fn with_options(options: GpuContextOptions) -> Result<Self, GpuError> {
        // Create instance
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: options.backends,
            ..Default::default()
        });

        // Request adapter
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: options.power_preference,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|_| GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        tracing::info!(
            adapter = %adapter_info.name,
            backend = ?adapter_info.backend,
            device_type = ?adapter_info.device_type,
            "GPU adapter selected"
        );

        // Request device
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Continuum GPU"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        ))
        .map_err(|e: wgpu::RequestDeviceError| GpuError::DeviceRequest(e.to_string()))?;

        Ok(Self {
            device,
            queue,
            adapter_info,
        })
    }

    /// Get adapter name for diagnostics.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Get adapter backend type.
    pub fn backend(&self) -> wgpu::Backend {
        self.adapter_info.backend
    }

    /// Get device type (discrete, integrated, etc.).
    pub fn device_type(&self) -> wgpu::DeviceType {
        self.adapter_info.device_type
    }

    /// Access the underlying wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Access the underlying wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Create a storage buffer for compute operations.
    pub fn create_storage_buffer(&self, label: &str, size: u64, read_only: bool) -> wgpu::Buffer {
        let usage = if read_only {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
        } else {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
        };

        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Create a staging buffer for CPU readback.
    pub fn create_staging_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Upload data to a buffer.
    pub fn upload_buffer<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, data: &[T]) {
        self.queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
    }

    /// Submit commands and wait for completion.
    pub fn submit_and_wait(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }
}

/// Options for GPU context creation.
#[derive(Debug, Clone)]
pub struct GpuContextOptions {
    /// Backend APIs to consider.
    pub backends: wgpu::Backends,
    /// Power preference for adapter selection.
    pub power_preference: wgpu::PowerPreference,
}

impl Default for GpuContextOptions {
    fn default() -> Self {
        Self {
            backends: wgpu::Backends::PRIMARY,
            power_preference: wgpu::PowerPreference::HighPerformance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a GPU and are marked as ignored by default.
    // Run with: cargo test --features gpu -- --ignored

    #[test]
    #[ignore = "requires GPU"]
    fn test_context_creation() {
        let ctx = GpuContext::new().unwrap();
        println!("Adapter: {}", ctx.adapter_name());
        println!("Backend: {:?}", ctx.backend());
        println!("Device type: {:?}", ctx.device_type());
    }

    #[test]
    #[ignore = "requires GPU"]
    fn test_buffer_creation() {
        let ctx = GpuContext::new().unwrap();

        let input_buffer = ctx.create_storage_buffer("test_input", 1024, true);
        let output_buffer = ctx.create_storage_buffer("test_output", 1024, false);
        let staging_buffer = ctx.create_staging_buffer("test_staging", 1024);

        // Buffers created successfully
        assert_eq!(input_buffer.size(), 1024);
        assert_eq!(output_buffer.size(), 1024);
        assert_eq!(staging_buffer.size(), 1024);
    }
}
