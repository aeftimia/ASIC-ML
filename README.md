# What Is This?

This is a pytorch model that learns to design digital circuits. The circuits are templated as an N dimensnional grid of binary inputs which are fed into a grid of logic gates with the same dimensions as the grid of inputs. Each logic gate takes nearby inputs and learns its own truth table. The outputs of these gates may be fed into several more layers of gates before the output is recovered. 

Input 1 | Input 2 | Input 3 | Input 4 |\
Gate  1 | Gate  2 | Gate  3 | Gate  4 |\
Gate' 1 | Gate' 2 | Gate' 3 | Gate' 4 |\
Out   1 | Out   2 | Out   3 | Out   4 |

The model takes a parameter `shape` which determines that shape of the N dimensional grid of inputs. In the above example, `shape=(4,)`, and `num_layers=4`. To borrow terminology from deep learning, the inputs are _convolved_ before being fed to a gate. In the above example, a kernel size of 3 would mean Gate 1 sees Inputs 4, 1, and 2. Gate 2 sees inputs 1, 2, and 3. Gate 3 sees inputs 2, 3, and 4. Gate 4 sees inputs 3, 4, and 1. Notice the convolution current wraps the grid. This was originally though to help the network learn to build modular adders (one of the first working examples), but may be subject to change in the future. The `kernel_offset` parameter can be used to determine which gates see which convolution. It can one of three values, `'center'`, `'left'`, `'right'`, and is `'center'` by default. `'right'` would mean Gate 1 sees inputs 3, 4, and 1, and `'left'` means Gate 1 would see inputs 1, 2, and 3. You can also specify a tuple of values with the same length as `shape` to set different offsets along different dimensions.

Inputs can be _embedded_ in a larger grid that consists of both inputs and memory bits. For example, if I build an ASIC that expects 4 inputs along one dimension, and I feed it an input of length 2, it will embed the input into larger grid that evenly disperses 2 "memory" bits (initially set to 0), and recover the appropriate 2 output bits.

Input 1 | 0       | Input 2 | 0       |\
Gate  1 | Gate  2 | Gate  3 | Gate  4 |\
Gate' 1 | Gate' 2 | Gate' 3 | Gate' 4 |\
Out   1 | Ignored | Out   2 | Ignored |

`weight_sharing` contrains gates along different dimensions to share the same truth tables. Weight sharing is set to `False` along all dimensions by default. 
`recure` determines the number of times the output of the entire circuit should be fed back into the input before being returned to the user.
