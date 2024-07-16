import triton 
import triton.language as tl 
import torch
import pdb

@triton.jit
def kernel_vector_addition(a_ptr, b_ptr, out_ptr,
                           num_elements: tl.constexpr,
                           block_size: tl.constexpr,
                           ):
    
    pid  = tl.program_id(0)
    tl.device_print("PID: ", pid)
    pdb.set_trace()
    block_start = pid * block_size
    thread_offsets = block_start + tl.arange(0, block_size)
    mask = thread_offsets < num_elements
    a_pointers = tl.load(a_ptr + thread_offsets, mask=mask)
    b_pointers = tl.load(b_ptr + thread_offsets, mask=mask)
    result = a_pointers + b_pointers
    tl.store(out_ptr + thread_offsets, result, mask=mask)


def ceil_div(x: int, y: int)-> int:
    return ((x+y)//y)


def vector_addition(a: torch.tensor, b: torch.tensor) -> torch.tensor:
    output_buffer = torch.empty_like(a)
    assert a.is_cuda and b.is_cuda
    num_elements = a.numel()
    assert num_elements == b.numel()

    block_size = 128
    grid_size = ceil_div(num_elements, block_size)

    grid = (grid_size,)

    k2 = kernel_vector_addition[grid](
        a, b, output_buffer, 
        num_elements, 
        block_size)

    return output_buffer


def main():
    ## verify numerical fidelity
    torch.manual_seed(2024)
    vec_size = 8192
    a = torch.rand(vec_size, device = "cuda")
    b = torch.rand_like(a)
    torch_res = a+b
    triton_res = vector_addition(a, b)
    fidelity = torch.allclose(torch_res, triton_res)
    print("Fidelity: ", fidelity)

if __name__ == "__main__":
    main()