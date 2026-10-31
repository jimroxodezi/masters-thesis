from rtlsdr import RtlSdr

sdr = RtlSdr()
sdr.sample_rate = 2.4e6   # Hz
sdr.center_freq = 100e6   # 100 MHz (FM band, easy to verify)
sdr.gain = 'auto'

samples = sdr.read_samples(256 * 1024)
sdr.close()

print(f"Captured {len(samples)} samples")
print(f"Sample dtype: {samples.dtype}")