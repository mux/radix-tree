use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use replace_with::replace_with_or_abort;

pub trait AsSlice<K> {
    fn as_slice(&self) -> &[K];
}

impl AsSlice<u8> for &str {
    fn as_slice(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl AsSlice<u8> for String {
    fn as_slice(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<K> AsSlice<K> for &[K] {
    fn as_slice(&self) -> &[K] {
        self
    }
}

impl<K> AsSlice<K> for Vec<K> {
    fn as_slice(&self) -> &[K] {
        self.as_slice()
    }
}

/// A Radix Tree implementation. Great attention has been given to performance, but this has not
/// yet been benchmarked, and the test coverage is quite low. Use at your own risk.
///
/// Children are represented using a vector, which is fine when the fan-out is low, but might be
/// suboptimal when it isn't. Future versions might provide alternative ways to represent children,
/// such as sorted vectors, sorted lists, or hash maps.
///
/// The main type is `RadixTree`, which maintains a count of the elements in the tree, allowing an
/// O(1) implementation of `len`. All methods on `RadixTreeNode` can be used on `RadixTree` through
/// the `Deref` and `DerefMut` implementations.
///
/// The API is modeled after the standard HashMap type.
#[derive(PartialEq, Eq, Debug)]
pub struct RadixTree<K, V>
where
    K: PartialEq,
{
    count: usize,
    root: RadixTreeNode<K, V>,
}

#[derive(PartialEq, Eq, Debug)]
pub struct RadixTreeNode<K, V>
where
    K: PartialEq,
{
    value: Option<V>,
    // Representing children as a vector of edges is very debatable. There are a lot of options
    // with different strengths and weaknesses. Using a HashMap would provide better performance to
    // locate relevant edges but the memory overhead could be significant in a data structure that
    // is very memory efficient by nature. A LinkedList would allow to remove elements in O(1)
    // without the need for moving elements around but would also have poor memory locality.
    // Maintaining a sorted vector (or list) would amortize prefix lookup at the cost of slightly
    // less efficient insertion, etc...
    edges: Vec<(Vec<K>, RadixTreeNode<K, V>)>,
}

impl<K, V> RadixTree<K, V>
where
    K: PartialEq + Copy,
{
    /// Creates an empty `RadixTree`.
    pub fn new() -> RadixTree<K, V> {
        RadixTree {
            count: 0,
            root: RadixTreeNode::new(),
        }
    }

    /// Creates a `RadixTree` with a single value at the root.
    pub fn singleton(value: V) -> RadixTree<K, V> {
        RadixTree {
            count: 1,
            root: RadixTreeNode::singleton(value),
        }
    }

    /// Returns the number of elements in the tree. This is O(1).
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if the tree contains no elements.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Tries to insert a key-value pair into the tree, and returns a mutable reference to the
    /// value in this entry.
    pub fn insert<T>(&mut self, key: T, value: V) -> Option<V>
    where
        T: AsSlice<K>,
    {
        let optv = self.root.insert(key, value);
        if optv.is_none() {
            self.count += 1;
        }
        optv
    }

    /// Removes a key from the tree, returning the stored key and value if the key was previously
    /// in the tree.
    pub fn remove<T>(&mut self, key: T) -> Option<V>
    where
        T: AsSlice<K>,
    {
        let optv = self.root.remove(key);
        if optv.is_some() {
            self.count -= 1;
        }
        optv
    }

    /// Clears the map, removing all key-value pairs.
    pub fn clear(&mut self) {
        self.root.clear();
        self.count = 0;
    }
}

impl<K, V> Deref for RadixTree<K, V>
where
    K: PartialEq,
{
    type Target = RadixTreeNode<K, V>;

    fn deref(&self) -> &Self::Target {
        &self.root
    }
}

impl<K, V> DerefMut for RadixTree<K, V>
where
    K: PartialEq,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.root
    }
}

impl<K: PartialEq + Copy, V> Default for RadixTree<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> RadixTreeNode<K, V>
where
    K: PartialEq + Copy,
{
    fn new() -> RadixTreeNode<K, V> {
        RadixTreeNode {
            value: None,
            edges: Vec::new(),
        }
    }

    fn singleton(value: V) -> RadixTreeNode<K, V> {
        RadixTreeNode {
            value: Some(value),
            edges: Vec::new(),
        }
    }

    /// Returns `true` if the node is a leaf node.
    pub fn is_leaf(&self) -> bool {
        self.edges.is_empty()
    }

    /// Returns `true` if the node is not a leaf node.
    pub fn is_node(&self) -> bool {
        !self.is_leaf()
    }

    /// Returns a reference to the value at the root node.
    pub fn value(&self) -> Option<&V> {
        self.get(&[] as &[K])
    }

    /// Returns the number of elements in a tree. Unlike the [`RadixTree`] implementation, this
    /// needs to traverse the full tree and is thus O(n).
    pub fn len(&self) -> usize {
        self.values().count()
    }

    /// Returns `true` if the tree contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if the tree contains a value for the specified key.
    pub fn contains_key<T>(&self, key: T) -> bool
    where
        T: AsSlice<K>,
    {
        self.get(key).is_some()
    }

    // All the iterators. We need into_iter() as well. The "edges" variants return the edge labels
    // and do not reconstruct the full keys. It can be useful for performance.
    pub fn iter<'a>(&'a self) -> Iter<'a, K, V> {
        Iter {
            node: self,
            prefix: Vec::new(),
            parents: Vec::new(),
            yielded: false,
            index: 0,
        }
    }

    pub fn keys<'a>(&'a self) -> Keys<'a, K, V> {
        Keys {
            node: self,
            parents: Vec::new(),
            prefix: Vec::new(),
            yielded: false,
            index: 0,
        }
    }

    pub fn edges<'a>(&'a self) -> Edges<'a, K, V> {
        Edges {
            node: self,
            parents: Vec::new(),
            prefix: &[],
            yielded: false,
            index: 0,
        }
    }

    pub fn values<'a>(&'a self) -> Values<'a, K, V> {
        Values {
            node: self,
            parents: Vec::new(),
            yielded: false,
            index: 0,
        }
    }

    pub fn iter_edges<'a>(&'a self) -> IterEdges<'a, K, V> {
        IterEdges {
            node: self,
            prefix: &[],
            parents: Vec::new(),
            yielded: false,
            index: 0,
        }
    }

    pub fn iter_mut<'a>(&'a mut self) -> IterMut<'a, K, V> {
        IterMut {
            node: std::ptr::from_mut(self),
            prefix: Vec::new(),
            parents: Vec::new(),
            yielded: false,
            index: 0,
            _marker: PhantomData,
        }
    }

    pub fn iter_edges_mut<'a>(&'a mut self) -> IterEdgesMut<'a, K, V> {
        IterEdgesMut {
            node: std::ptr::from_mut(self),
            prefix: &[],
            parents: Vec::new(),
            yielded: false,
            index: 0,
            _marker: PhantomData,
        }
    }

    pub fn values_mut<'a>(&'a mut self) -> ValuesMut<'a, K, V> {
        ValuesMut {
            node: std::ptr::from_mut(self),
            parents: Vec::new(),
            yielded: false,
            index: 0,
            _marker: PhantomData,
        }
    }

    /// Returns the subtree matching the given prefix.
    pub fn at_prefix<T>(&self, key: T) -> Option<&RadixTreeNode<K, V>>
    where
        T: AsSlice<K>,
    {
        let key = key.as_slice();
        if key.is_empty() {
            return Some(self);
        }
        for (prefix, child) in &self.edges {
            if let Some(rest) = key.strip_prefix(prefix.as_slice()) {
                return child.at_prefix(rest);
            }
        }
        None
    }

    /// Mutable variant of [`at_prefix`].
    ///
    /// [`at_prefix`]: RadixTreeNode::at_prefix
    pub fn at_prefix_mut<T>(&mut self, key: T) -> Option<&mut RadixTreeNode<K, V>>
    where
        T: AsSlice<K>,
    {
        let key = key.as_slice();
        if key.is_empty() {
            return Some(self);
        }
        for (prefix, child) in &mut self.edges {
            if let Some(rest) = key.strip_prefix(prefix.as_slice()) {
                return child.at_prefix_mut(rest);
            }
        }
        None
    }

    /// Returns a reference to the value corresponding to the key.
    pub fn get<T>(&self, key: T) -> Option<&V>
    where
        T: AsSlice<K>,
    {
        let key = key.as_slice();
        self.at_prefix(key).and_then(|node| node.value.as_ref())
    }

    /// Returns a mutable reference to the value corresponding to the key.
    pub fn get_mut<T>(&mut self, key: T) -> Option<&mut V>
    where
        T: AsSlice<K>,
    {
        self.at_prefix_mut(key).and_then(|node| node.value.as_mut())
    }

    fn insert<T>(&mut self, key: T, value: V) -> Option<V>
    where
        T: AsSlice<K>,
    {
        let key = key.as_slice();
        if key.is_empty() {
            return self.value.replace(value);
        }
        for (prefix, child) in &mut self.edges {
            let common_len = longest_common_prefix(prefix, key);
            if common_len > 0 {
                if common_len == prefix.len() {
                    return child.insert(&key[common_len..], value);
                }
                // We need to split the node.
                let prefix_rest = prefix.drain(common_len..).collect();
                replace_with_or_abort(child, |node| RadixTreeNode {
                    value: None,
                    edges: vec![
                        (prefix_rest, node),
                        (key[common_len..].to_vec(), RadixTreeNode::singleton(value)),
                    ],
                });
                return None;
            }
        }
        self.edges
            .push((key.to_vec(), RadixTreeNode::singleton(value)));
        None
    }

    fn remove<T>(&mut self, key: T) -> Option<V>
    where
        T: AsSlice<K>,
    {
        let key = key.as_slice();
        if key.is_empty() {
            return self.value.take();
        }
        let mut cleanup_node = None;
        for (i, (prefix, child)) in self.edges.iter_mut().enumerate() {
            let common_len = longest_common_prefix(prefix, key);
            if common_len > 0 {
                if common_len == prefix.len() {
                    let removed = child.remove(&key[common_len..]);
                    // If something was removed and the child node doesn't hold a value, we might need to do some cleanup.
                    if removed.is_some() && child.value.is_none() {
                        if child.edges.is_empty() {
                            cleanup_node = Some((i, removed));
                            break;
                        }
                        if child.edges.len() == 1 {
                            let (child_prefix, grandchild) = child.edges.remove(0);
                            prefix.extend(child_prefix);
                            *child = grandchild;
                        }
                    }
                    return removed;
                }
                // This edge and our key have a common prefix but the key does not match the
                // entire edge. Since no other edge can match our key if the tree is well
                // formed, this means that there is no such key in our tree.
                return None;
            }
        }
        if let Some((i, removed)) = cleanup_node {
            self.edges.remove(i);
            return removed;
        }
        None
    }

    fn clear(&mut self) {
        self.value.take();
        self.edges.clear();
    }
}

impl<K: PartialEq + Copy, V> Default for RadixTreeNode<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Iter<'a, K, V>
where
    K: PartialEq,
{
    parents: Vec<(&'a RadixTreeNode<K, V>, usize, usize)>,
    node: &'a RadixTreeNode<K, V>,
    prefix: Vec<K>,
    yielded: bool,
    index: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V>
where
    K: PartialEq + Clone,
{
    type Item = (Vec<K>, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if !self.yielded
                && let Some(val) = &self.node.value
            {
                self.yielded = true;
                return Some((self.prefix.clone(), val));
            }
            if let Some((prefix, node)) = self.node.edges.get(self.index) {
                self.parents
                    .push((self.node, self.index + 1, self.prefix.len()));
                self.node = node;
                self.prefix.extend(prefix.iter().cloned());
                self.yielded = false;
                self.index = 0;
            } else if let Some((node, index, prefix_len)) = self.parents.pop() {
                self.prefix.truncate(prefix_len);
                self.node = node;
                self.index = index;
                self.yielded = true;
            } else {
                return None;
            }
        }
    }
}

pub struct Keys<'a, K, V>
where
    K: PartialEq,
{
    parents: Vec<(&'a RadixTreeNode<K, V>, usize, usize)>,
    node: &'a RadixTreeNode<K, V>,
    prefix: Vec<K>,
    yielded: bool,
    index: usize,
}

impl<'a, K, V> Iterator for Keys<'a, K, V>
where
    K: PartialEq + Clone,
{
    type Item = Vec<K>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if !self.yielded && self.node.value.is_some() {
                self.yielded = true;
                return Some(self.prefix.clone());
            }
            if let Some((prefix, node)) = self.node.edges.get(self.index) {
                self.parents
                    .push((self.node, self.index + 1, self.prefix.len()));
                self.node = node;
                self.prefix.extend(prefix.iter().cloned());
                self.yielded = false;
                self.index = 0;
            } else if let Some((node, index, prefix_len)) = self.parents.pop() {
                self.prefix.truncate(prefix_len);
                self.node = node;
                self.index = index;
                self.yielded = true;
            } else {
                return None;
            }
        }
    }
}

pub struct Values<'a, K, V>
where
    K: PartialEq,
{
    parents: Vec<(&'a RadixTreeNode<K, V>, usize)>,
    node: &'a RadixTreeNode<K, V>,
    yielded: bool,
    index: usize,
}

impl<'a, K, V> Iterator for Values<'a, K, V>
where
    K: PartialEq,
{
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if !self.yielded
                && let Some(val) = &self.node.value
            {
                self.yielded = true;
                return Some(val);
            }
            if let Some((_, node)) = self.node.edges.get(self.index) {
                self.parents.push((self.node, self.index + 1));
                self.node = node;
                self.yielded = false;
                self.index = 0;
            } else if let Some((node, index)) = self.parents.pop() {
                self.node = node;
                self.index = index;
                self.yielded = true;
            } else {
                return None;
            }
        }
    }
}

pub struct IterEdges<'a, K, V>
where
    K: PartialEq,
{
    parents: Vec<(&'a RadixTreeNode<K, V>, usize)>,
    node: &'a RadixTreeNode<K, V>,
    prefix: &'a [K],
    yielded: bool,
    index: usize,
}

impl<'a, K, V> Iterator for IterEdges<'a, K, V>
where
    K: PartialEq,
{
    type Item = (&'a [K], &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if !self.yielded
                && let Some(val) = &self.node.value
            {
                self.yielded = true;
                return Some((self.prefix, val));
            }
            if let Some((prefix, node)) = self.node.edges.get(self.index) {
                self.parents.push((self.node, self.index + 1));
                self.node = node;
                self.prefix = prefix.as_slice();
                self.yielded = false;
                self.index = 0;
            } else if let Some((node, index)) = self.parents.pop() {
                self.node = node;
                self.index = index;
                self.yielded = true;
            } else {
                return None;
            }
        }
    }
}

pub struct Edges<'a, K, V>
where
    K: PartialEq,
{
    parents: Vec<(&'a RadixTreeNode<K, V>, usize)>,
    node: &'a RadixTreeNode<K, V>,
    prefix: &'a [K],
    yielded: bool,
    index: usize,
}

impl<'a, K, V> Iterator for Edges<'a, K, V>
where
    K: PartialEq,
{
    type Item = &'a [K];

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if !self.yielded && self.node.value.is_some() {
                self.yielded = true;
                return Some(self.prefix);
            }
            if let Some((prefix, node)) = self.node.edges.get(self.index) {
                self.parents.push((self.node, self.index + 1));
                self.node = node;
                self.prefix = prefix.as_slice();
                self.yielded = false;
                self.index = 0;
            } else if let Some((node, index)) = self.parents.pop() {
                self.node = node;
                self.index = index;
                self.yielded = true;
            } else {
                return None;
            }
        }
    }
}

pub struct IterMut<'a, K, V>
where
    K: PartialEq,
{
    parents: Vec<(*mut RadixTreeNode<K, V>, usize, usize)>,
    node: *mut RadixTreeNode<K, V>,
    prefix: Vec<K>,
    yielded: bool,
    index: usize,
    _marker: PhantomData<&'a mut V>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V>
where
    K: PartialEq + Clone,
{
    type Item = (Vec<K>, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node = unsafe { &mut *self.node };
            if !self.yielded
                && let Some(val) = &mut node.value
            {
                self.yielded = true;
                return Some((self.prefix.clone(), val));
            }
            if let Some((prefix, node)) = node.edges.get_mut(self.index) {
                self.parents
                    .push((self.node, self.index + 1, self.prefix.len()));
                self.node = node;
                self.prefix.extend(prefix.iter().cloned());
                self.yielded = false;
                self.index = 0;
            } else if let Some((node, index, prefix_len)) = self.parents.pop() {
                self.prefix.truncate(prefix_len);
                self.node = node;
                self.index = index;
                self.yielded = true;
            } else {
                return None;
            }
        }
    }
}

pub struct IterEdgesMut<'a, K, V>
where
    K: PartialEq,
{
    parents: Vec<(*mut RadixTreeNode<K, V>, usize)>,
    node: *mut RadixTreeNode<K, V>,
    prefix: &'a [K],
    yielded: bool,
    index: usize,
    _marker: PhantomData<&'a mut V>,
}

impl<'a, K, V> Iterator for IterEdgesMut<'a, K, V>
where
    K: PartialEq,
{
    type Item = (&'a [K], &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node = unsafe { &mut *self.node };
            if !self.yielded
                && let Some(val) = &mut node.value
            {
                self.yielded = true;
                return Some((self.prefix, val));
            }
            if let Some((prefix, node)) = node.edges.get_mut(self.index) {
                self.parents.push((self.node, self.index + 1));
                self.node = node;
                self.prefix = prefix.as_slice();
                self.yielded = false;
                self.index = 0;
            } else if let Some((node, index)) = self.parents.pop() {
                self.node = node;
                self.index = index;
                self.yielded = true;
            } else {
                return None;
            }
        }
    }
}

pub struct ValuesMut<'a, K, V>
where
    K: PartialEq,
{
    parents: Vec<(*mut RadixTreeNode<K, V>, usize)>,
    node: *mut RadixTreeNode<K, V>,
    yielded: bool,
    index: usize,
    _marker: PhantomData<&'a mut V>,
}

impl<'a, K, V> Iterator for ValuesMut<'a, K, V>
where
    K: PartialEq,
{
    type Item = &'a mut V;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node = unsafe { &mut *self.node };
            if !self.yielded
                && let Some(val) = &mut node.value
            {
                self.yielded = true;
                return Some(val);
            }
            if let Some((_, node)) = node.edges.get_mut(self.index) {
                self.parents.push((self.node, self.index + 1));
                self.node = node;
                self.yielded = false;
                self.index = 0;
            } else if let Some((node, index)) = self.parents.pop() {
                self.node = node;
                self.index = index;
                self.yielded = true;
            } else {
                return None;
            }
        }
    }
}
fn longest_common_prefix<T>(s1: &[T], s2: &[T]) -> usize
where
    T: PartialEq,
{
    s1.iter()
        .zip(s2.iter())
        .position(|(x, y)| x != y)
        .unwrap_or_else(|| s1.len().min(s2.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_common_prefix_works() {
        assert_eq!(longest_common_prefix(b"bar", b"baz"), 2);
        assert_eq!(longest_common_prefix(b"bar", b"barbie"), 3);
        assert_eq!(longest_common_prefix(b"foo", b"bar"), 0);
        assert_eq!(longest_common_prefix(b"foo", b"foo"), 3);
    }

    #[test]
    fn radix_tree_works() {
        let mut tree = RadixTree::new();
        assert_eq!(tree.value, None);
        assert_eq!(tree.insert("foo", 42), None);
        assert_eq!(tree.value, None);
        assert_eq!(tree.edges.len(), 1);
        let _node = tree.at_prefix("foo");
        assert_eq!(_node.and_then(|node| node.value), Some(42));
        assert!(_node.is_some_and(|node| node.edges.is_empty()));

        assert!(tree.insert("bar", 13).is_none());
        assert_eq!(tree.get("bar"), Some(&13));
        assert!(tree.insert("baz", 7).is_none());
        assert_eq!(tree.get("baz"), Some(&7));
        // This should have split the "bar" node into a "ba" node with one "r" edge and one "z"
        // edge with the values of "bar" and "baz" respectively.
        let _node = tree.at_prefix("ba");
        assert!(_node.is_some_and(|node| node.value.is_none()));
        assert!(_node.is_some_and(|node| node.edges.len() == 2));
        assert!(_node.is_some_and(|node| {
            let (prefix, child) = &node.edges[0];
            prefix == b"r" && child.value == Some(13)
        }));
        assert!(_node.is_some_and(|node| {
            let (prefix, child) = &node.edges[1];
            prefix == b"z" && child.value == Some(7)
        }));

        assert_eq!(tree.insert("ba", 18), None);
        assert_eq!(tree.get("ba"), Some(&18));
        assert_eq!(tree.insert("barbie", 23), None);
        assert_eq!(tree.get("barbie"), Some(&23));
        assert_eq!(tree.get("bag"), None);
        assert_eq!(tree.get("qux"), None);
        assert_eq!(tree.insert("ba", 27), Some(18));
        assert_eq!(tree.get("ba"), Some(&27));

        println!("Keys matching prefix \"ba\" and their values");
        let subtree = tree.at_prefix("ba").unwrap();
        for (key, value) in subtree.iter() {
            let key = unsafe { String::from_utf8_unchecked(key.to_vec()) };
            println!("\"{key}\": {value}");
        }

        println!("All values");
        for v in tree.values() {
            println!("{v}");
        }

        println!("All keys and values");
        for (key, val) in tree.iter() {
            let key = unsafe { String::from_utf8_unchecked(key.to_vec()) };
            println!("\"{key}\": {val}");
        }

        println!("Fully reconstructed keys");
        for key in tree.keys() {
            let key = unsafe { String::from_utf8_unchecked(key.to_vec()) };
            println!("\"{key}\"");
        }

        println!("Incrementing all values by 1");
        for (key, val) in tree.iter_mut() {
            let key = unsafe { String::from_utf8_unchecked(key.to_vec()) };
            *val += 1;
            println!("\"{key}\": {val}");
        }

        assert_eq!(tree.remove("bar"), Some(14));
        println!("{tree:?}");
        // XXX Check that we now have "ba" -> "rbie" and "ba" -> "z"
        assert_eq!(tree.get("bar"), None);
        assert_eq!(tree.remove("baz"), Some(8));
        assert_eq!(tree.remove("baz"), None);
    }
}
