# grad
A differentiation library

train on github runners!!! Upload artifacts as release of package of repo

ml frameworks are computational graphs of tensor and operator nodes with
subgraph matching to fused kernels

todo:
1. **Reshape/View** (as discussed) - Fundamental for attention mechanisms and tensor manipulation

2. **Transpose/Permute** - Cannot be decomposed, essential for:
   - Switching batch and sequence dimensions
   - Rearranging attention heads
   - Matrix operations in general

3. **Slice/Index/Gather** - Fundamental for:
   - Selecting specific tokens or positions
   - Attention mask application
   - Embedding lookups (essential for token embeddings)

4. **Concatenation** - Fundamental for:
   - Combining multiple attention heads
   - Joining sequences
   - Cannot be decomposed into other operations