use crate::dinic::DinicGraph;
use rand::prelude::SliceRandom;
use rand::Rng;
use serde::Deserialize;
use std::{fs::File, path::Path};

#[derive(Deserialize, Debug)]
pub struct Graph {
    size: usize,
    edges: Box<[Box<[usize]>]>,
}

struct ShuffledDinicGraph {
    graph: DinicGraph,
    left: Box<[usize]>,
    right: Box<[usize]>,
}

impl ShuffledDinicGraph {
    pub fn current_flow(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.graph
            .current_flow()
            .filter(|x| x.0 < self.left.len())
            .map(|(u, v, _)| (self.left[u], self.right[v - self.right.len()]))
    }
}

impl Graph {
    fn load<S: AsRef<Path>>(x: S) -> anyhow::Result<Self> {
        let file = File::open(x)?;
        simd_json::from_reader(file).map_err(Into::into)
    }
    pub fn as_shuffled_dinic_graph<G: Rng>(&self, rng: &mut G) -> ShuffledDinicGraph {
        let src = 2 * self.size;
        let sink = 2 * self.size + 1;
        let mut graph = DinicGraph::new(2 * self.size + 2, src, sink);
        for i in 0..self.size {
            graph.add_edge(src, i, 1);
            graph.add_edge(i + self.size, sink, 1);
        }
        let mut left_shuffle = (0..self.size).collect::<Box<[usize]>>();
        let mut right_shuffle = (0..self.size).collect::<Box<[usize]>>();
        left_shuffle.shuffle(rng);
        right_shuffle.shuffle(rng);
        for (u, edges) in self.edges.iter().enumerate() {
            for v in edges.iter().copied() {
                graph.add_edge(left_shuffle[u], right_shuffle[v] + self.size, 1);
            }
        }
        let mut left_decode = vec![0; self.size].into_boxed_slice();
        let mut right_decode = vec![0; self.size].into_boxed_slice();
        for (i, &x) in left_shuffle.iter().enumerate() {
            left_decode[x] = i;
        }
        for (i, &x) in right_shuffle.iter().enumerate() {
            right_decode[x] = i;
        }
        ShuffledDinicGraph {
            graph,
            left: left_decode,
            right: right_decode,
        }
    }
}

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    #[test]
    fn box_example() {
        let path: PathBuf = env!("PWD").into();
        let path = path.join("data").join("box.json");
        let graph = super::Graph::load(path).unwrap();
        println!("{:?}", graph);
        let mut dinic = graph.as_shuffled_dinic_graph(&mut rand::thread_rng());
        let flow = dinic.graph.calculate_flow();
        println!("flow: {}", flow);
        assert_eq!(flow as usize, graph.size);
        for (u, v) in dinic.current_flow() {
            println!("{} -> {}", u, v);
            assert_eq!(u, v);
        }
    }

    #[test]
    fn complete_example() {
        let path: PathBuf = env!("PWD").into();
        let path = path.join("data").join("complete.json");
        let graph = super::Graph::load(path).unwrap();
        println!("{:?}", graph);
        let mut dinic = graph.as_shuffled_dinic_graph(&mut rand::thread_rng());
        let flow = dinic.graph.calculate_flow();
        println!("flow: {}", flow);
        assert_eq!(flow as usize, graph.size);
        for (u, v) in dinic.current_flow() {
            println!("{} -> {}", u, v);
        }
    }
}
