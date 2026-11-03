"""
occupancy
---------
LTE downlink occupancy detection with an RTL-SDR Blog V4: capture and survey
tools, preprocessing, autoencoder and classical detectors, calibration at fixed
false-alarm rates, session-level evaluation, latency and a real-time demo.
The autoencoders train on the laptop's CPU or on a CUDA GPU, for example on
Google Colab from a training pack (pack.py), with checkpoints that survive
interruptions (train.py); the trained models are imported back with unpack.py.

The experiment is specified in notes/method-design.md; every setting lives in
occupancy/config.py. Command line: python -m occupancy --help
"""

__version__ = "0.3.0"
