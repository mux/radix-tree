A Radix Tree implementation. Great attention has been given to performance, but this has not
yet been benchmarked, and the test coverage is quite low. Use at your own risk.

Children are represented using a vector, which is fine when the fan-out is low, but might be
suboptimal when it isn't. Future versions might provide alternative ways to represent children,
such as sorted vectors, sorted lists, or hash maps.

The main type is `RadixTree`, which maintains a count of the elements in the tree, allowing an
O(1) implementation of `len`. All methods on `RadixTreeNode` can be used on `RadixTree` through
the `Deref` and `DerefMut` implementations.

The API is modeled after the standard HashMap type.

# TODO

- Children should be sorted
- Add more tests
- Benchmarks
- Consuming iterator
- Entry API
