# llama3_inference_toy

The core idea is the understanding of llama 3 architecture and how to speed-up inference using techniques like kernel engineering and quantization.

The techniques implemented here are far from production grade and are meant for educational purposes only, as some necesary dequantizations are not implemented within the proper kernels (yet).

We will use Torchao's W8A16 quantization paradigm, where weights are stored in 8 bit and activations are stored in 16 bit. To achieve inference speedup we will use Triton, which will allow us to write some custom kernels that manage core parts of the model architecture at the same time that handles quantization and dequantization, memory management, etc.

This is a block of llama 3 8b architecture:

The idea is to implement the RMSNorm operation + dynamic quantization in the same kernel. We will monkey patch the kernel and use torch to handle operations we do not want to implement ourselves inside other kernels (for example int8 matmul, which we will use, or dequantization after attention and swiglu blocks).

![alt text](<transformer llama block w8a8(1).png>)
