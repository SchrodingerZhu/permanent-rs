use crate::dinic::DinicGraph;
use serde::Deserialize;
use std::{fs::File, path::Path};

#[derive(Deserialize, Debug)]
pub struct Graph {
    size: usize,
    edges: Box<[Box<[usize]>]>,
}

pub struct Match {
    edges: Box<[(usize, usize)]>,
}

impl Match {
    pub fn size(&self) -> usize {
        self.edges.len()
    }
}

impl Graph {
    fn load<S: AsRef<Path>>(x: S) -> anyhow::Result<Self> {
        let file = File::open(x)?;
        simd_json::from_reader(file).map_err(Into::into)
    }
    pub fn find_match(&self) -> Match {
        let src = 2 * self.size;
        let sink = 2 * self.size + 1;
        let mut graph = DinicGraph::new(2 * self.size + 2, src, sink);
        for i in 0..self.size {
            graph.add_edge(src, i, 1);
            graph.add_edge(i + self.size, sink, 1);
        }
        for (u, edges) in self.edges.iter().enumerate() {
            for v in edges.iter().copied() {
                graph.add_edge(u, v + self.size, 1);
            }
        }
        let flow = graph.calculate_flow();
        let mut edges = Vec::with_capacity(flow as usize);
        for (u, v, _) in graph.current_flow().filter(|x|x.0 < self.size) {
            edges.push((u, v - self.size));
        }
        Match {
            edges: edges.into_boxed_slice(),
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
        let matching = graph.find_match();
        assert_eq!(matching.size(), graph.size);
        for (u, v) in matching.edges.iter() {
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
        let matching = graph.find_match();
        assert_eq!(matching.size(), graph.size);
        for (u, v) in matching.edges.iter() {
            println!("{} -> {}", u, v);
        }
    }
}
