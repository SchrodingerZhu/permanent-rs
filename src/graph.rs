use crate::dinic::DinicGraph;
use serde::Deserialize;
use std::{fs::File, path::Path};

#[derive(Deserialize, Debug)]
pub struct Graph {
    size: usize,
    edges: Box<[Box<[usize]>]>,
}

impl Graph {
    fn load<S: AsRef<Path>>(x: S) -> anyhow::Result<Self> {
        let file = File::open(x)?;
        simd_json::from_reader(file).map_err(Into::into)
    }
    pub fn as_dinic_graph(&self) -> DinicGraph {
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
        graph
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
        let mut dinic = graph.as_dinic_graph();
        let flow = dinic.calculate_flow();
        println!("flow: {}", flow);
        assert_eq!(flow as usize, graph.size);
        for (u, v, _) in dinic.current_flow().filter(|x| x.0 < graph.size) {
            println!("{} -> {}", u, v - graph.size);
            assert_eq!(u, v - graph.size);
        }
    }

    #[test]
    fn complete_example() {
        let path: PathBuf = env!("PWD").into();
        let path = path.join("data").join("complete.json");
        let graph = super::Graph::load(path).unwrap();
        println!("{:?}", graph);
        let mut dinic = graph.as_dinic_graph();
        let flow = dinic.calculate_flow();
        println!("flow: {}", flow);
        assert_eq!(flow as usize, graph.size);
        for (u, v, _) in dinic.current_flow().filter(|x| x.0 < graph.size) {
            println!("{} -> {}", u, v - graph.size);
        }
    }
}
