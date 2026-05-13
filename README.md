# llama3_inference_toy

The core idea is the understanding of llama 3 architecture and how to speed-up inference using techniques like kernel engineering and quantization.

The techniques implemented here are far from production grade and are meant for educational purposes only, as some necesary dequantizations are not implemented within existing kernels (yet).

We will use Torchao's W8A16 quantization paradigm, where weights are stored in 8 bit and activations are stored in 16 bit. To achieve inference speedup we will use Triton, which will allow us to write some custom kernels that manage core parts of the model architecture at the same time that handles quantization and dequantization, memory management, etc.
