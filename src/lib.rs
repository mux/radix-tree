/// A Radix Tree implementation. Written mostly for fun and to learn Rust. Probably sub-optimal in
/// various ways, although I tried to reduce allocations and copies as much as possible by using
/// the replace_with crate, as well as Option::take() and Option::replace() as appropriate.
/// Hopefully not too buggy - there are a few tests but they don't even come close to covering
/// every invariant.
use replace_with::*;

#[derive(PartialEq, Eq, Debug)]
pub struct RadixTree<K, V>
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
    edges: Vec<(Vec<K>, RadixTree<K, V>)>,
}

impl<K, V> RadixTree<K, V>
where
    K: PartialEq + Copy,
{
    pub fn new() -> RadixTree<K, V> {
        RadixTree {
            value: None,
            edges: Vec::new(),
        }
    }

    pub fn with_value(value: V) -> RadixTree<K, V> {
        RadixTree {
            value: Some(value),
            edges: Vec::new(),
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.edges.is_empty()
    }

    pub fn is_node(&self) -> bool {
        !self.is_leaf()
    }

    pub fn value(&self) -> Option<&V> {
        self.get(&[])
    }

    // We could cache the number of elements at the root so this is O(1) but on the other hand,
    // the caller could easily maintain the length themselves.
    pub fn len(&self) -> usize {
        self.values().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        self.value.take();
        self.edges.clear();
    }

    // A bunch of iterators for various needs. We probably also need add keys_with(),
    // keys_with_fast(), iter_with_fast() and into_iter() variants as well.

    // I wonder if there's a better way to write this.
    pub fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = (Vec<K>, &'a V)> + 'a> {
        Box::new(
            self.value
                .as_ref()
                .map(|value| (Vec::new(), value))
                .into_iter()
                .chain(self.edges.iter().flat_map(|(prefix, node)| {
                    node.iter().map(|(inner_prefix, value)| {
                        let mut prefix = prefix.clone();
                        prefix.extend(inner_prefix);
                        (prefix, value)
                    })
                })),
        )
    }

    // Like iter() but we don't construct the full keys - it can be useful for performance.
    pub fn iter_fast<'a>(&'a self) -> Box<dyn Iterator<Item = (Vec<K>, &'a V)> + 'a> {
        Box::new(
            self.value
                .as_ref()
                .map(|value| (Vec::new(), value))
                .into_iter()
                .chain(self.edges.iter().flat_map(|(_, node)| node.iter_fast())),
        )
    }

    pub fn keys<'a>(&'a self) -> Box<dyn Iterator<Item = Vec<K>> + 'a> {
        Box::new(self.iter().map(|(key, _)| key))
    }

    pub fn keys_fast<'a>(&'a self) -> Box<dyn Iterator<Item = Vec<K>> + 'a> {
        Box::new(self.iter_fast().map(|(key, _)| key))
    }

    pub fn values<'a>(&'a self) -> Box<dyn Iterator<Item = &'a V> + 'a> {
        Box::new(self.iter_fast().map(|(_, value)| value))
    }

    pub fn values_with<'a>(&'a self, key: &[K]) -> Box<dyn Iterator<Item = &'a V> + 'a> {
        if key.is_empty() {
            return self.values();
        }
        for (prefix, child) in &self.edges {
            if let Some(rest) = key.strip_prefix(prefix.as_slice()) {
                return child.values_with(rest);
            }
        }
        Box::new(std::iter::empty())
    }

    /// All the keys matching the given prefix and their values. Keys do not include the prefix for
    /// performance reasons. The caller can reconstruct the full keys by prepending the prefix they
    /// passed as a parameter if needed.
    pub fn iter_with<'a>(&'a self, key: &[K]) -> Box<dyn Iterator<Item = (Vec<K>, &'a V)> + 'a> {
        if key.is_empty() {
            return self.iter();
        }
        for (prefix, child) in &self.edges {
            if let Some(rest) = key.strip_prefix(prefix.as_slice()) {
                return child.iter_with(rest);
            }
        }
        Box::new(std::iter::empty())
    }

    // Internal method returning the node for a given key. Used for testing and to implement get().
    fn lookup(&self, key: &[K]) -> Option<&RadixTree<K, V>> {
        if key.is_empty() {
            return Some(self);
        }
        for (prefix, child) in &self.edges {
            if let Some(rest) = key.strip_prefix(prefix.as_slice()) {
                return child.lookup(rest);
            }
        }
        None
    }

    // Hate this duplication but haven't yet found a way to avoid it.
    fn lookup_mut(&mut self, key: &[K]) -> Option<&mut RadixTree<K, V>> {
        if key.is_empty() {
            return Some(self);
        }
        for (prefix, child) in &mut self.edges {
            if let Some(rest) = key.strip_prefix(prefix.as_slice()) {
                return child.lookup_mut(rest);
            }
        }
        None
    }

    // Writing this in terms of lookup() hopefully doesn't hurt performance.
    pub fn get(&self, key: &[K]) -> Option<&V> {
        self.lookup(key).and_then(|node| node.value.as_ref())
    }

    pub fn get_mut(&mut self, key: &[K]) -> Option<&mut V> {
        self.lookup_mut(key).and_then(|node| node.value.as_mut())
    }

    pub fn insert(&mut self, key: &[K], value: V) -> Option<V> {
        if key.is_empty() {
            return self.value.replace(value);
        }
        for (prefix, child) in &mut self.edges {
            let common_len = longest_common_prefix(prefix, key);
            if common_len > 0 {
                if common_len == prefix.len() {
                    return child.insert(&key[common_len..], value);
                } else {
                    // We need to split the node.
                    //split_at = Some((i, common_len));
                    let prefix_rest = prefix.drain(common_len..).collect();
                    replace_with_or_abort(child, |node| RadixTree {
                        value: None,
                        edges: vec![
                            (prefix_rest, node),
                            (key[common_len..].to_vec(), RadixTree::with_value(value)),
                        ],
                    });
                    return None;
                }
            }
        }
        self.edges
            .push((key.to_vec(), RadixTree::with_value(value)));
        None
    }

    pub fn remove(&mut self, key: &[K]) -> Option<V> {
        if key.is_empty() {
            return self.value.take();
        }
        let mut cleanup_at = None;
        for (i, (prefix, child)) in self.edges.iter_mut().enumerate() {
            let common_len = longest_common_prefix(prefix, key);
            if common_len > 0 {
                if common_len == prefix.len() {
                    let removed = child.remove(&key[common_len..]);
                    // If something was removed and the child node doesn't hold a value, we might need to do some cleanup.
                    if removed.is_some() && child.value.is_none() {
                        cleanup_at = Some((i, removed));
                        break;
                    }
                    return removed;
                } else {
                    // This edge and our key have a common prefix but the key does not match the
                    // entire edge. Since no other edge can match our key if the tree is well
                    // formed, this means that there is no such key in our tree.
                    return None;
                }
            }
        }
        if let Some((i, removed)) = cleanup_at {
            unsafe {
                let (_, child) = self.edges.get_unchecked(i);
                if child.edges.is_empty() {
                    self.edges.remove(i);
                } else if child.edges.len() == 1 {
                    let (prefix, child) = self.edges.get_unchecked_mut(i);
                    let (child_prefix, grandchild) = child.edges.remove(0);
                    prefix.extend(child_prefix);
                    *child = grandchild;
                }
            }
            return removed;
        }
        None
    }
}

impl<K: PartialEq + Copy, V> Default for RadixTree<K, V> {
    fn default() -> Self {
        Self::new()
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
    }

    #[test]
    fn it_works() {
        let mut tree = RadixTree::new();
        assert!(tree.value.is_none());
        tree.insert(b"foo", 42);
        assert!(tree.value.is_none());
        assert_eq!(tree.edges.len(), 1);
        let _node = tree.lookup(b"foo");
        assert_eq!(_node.and_then(|node| node.value), Some(42));
        assert!(_node.is_some_and(|node| node.edges.is_empty()));

        tree.insert(b"bar", 13);
        assert_eq!(tree.get(b"bar"), Some(&13));
        tree.insert(b"baz", 7);
        assert_eq!(tree.get(b"baz"), Some(&7));
        // This should have split the "bar" node into a "ba" node with one "r" edge and one "z"
        // edge with the values of "bar" and "baz" respectively.
        let _node = tree.lookup(b"ba");
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

        tree.insert(b"ba", 18);
        assert_eq!(tree.get(b"ba"), Some(&18));
        tree.insert(b"barbie", 23);
        assert_eq!(tree.get(b"barbie"), Some(&23));
        assert!(tree.get(b"bag").is_none());
        assert!(tree.get(b"qux").is_none());
        tree.insert(b"ba", 27);
        assert_eq!(tree.get(b"ba"), Some(&27));

        println!("Keys matching prefix \"ba\" and their values");
        for kv in tree.iter_with(b"ba") {
            println!("{kv:?}");
        }

        println!("All values");
        for v in tree.values() {
            println!("{v}");
        }

        println!("All keys and values");
        for kv in tree.iter() {
            println!("{kv:?}");
        }

        assert_eq!(tree.remove(b"bar"), Some(13));
        println!("{tree:?}");
        // XXX Check that we now have "ba" -> "rbie" and "ba" -> "z""
        assert!(tree.get(b"bar").is_none());
        assert_eq!(tree.remove(b"baz"), Some(7));
        assert!(tree.remove(b"baz").is_none());
    }
}
