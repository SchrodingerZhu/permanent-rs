use std::collections::VecDeque;

struct Edge {
    points: (usize, usize),
    cap: isize,
    flow: isize,
}
pub struct DinicGraph {
    /// current flow
    flow: isize,
    /// map from node to edges
    adjacency: Vec<Vec<usize>>,
    /// edge storage
    edges: Vec<Edge>,
    /// level graph
    level: Vec<usize>,
    /// pointer to the next edge
    pointer: Vec<usize>,
    /// source node
    source: usize,
    /// sink node
    sink: usize,
}

impl DinicGraph {
    pub fn new(n: usize, source: usize, sink: usize) -> Self {
        DinicGraph {
            flow: 0,
            adjacency: vec![vec![]; n],
            edges: vec![],
            level: vec![usize::MAX; n],
            pointer: vec![0; n],
            source,
            sink,
        }
    }

    fn reset(&mut self) {
        self.level.fill(usize::MAX);
        self.pointer.fill(0);
    }

    pub fn add_edge(&mut self, from: usize, to: usize, cap: isize) {
        let m = self.edges.len();
        self.edges.push(Edge {
            points: (from, to),
            cap,
            flow: 0,
        });
        self.edges.push(Edge {
            points: (to, from),
            cap: 0,
            flow: 0,
        });
        self.adjacency[from].push(m);
        self.adjacency[to].push(m + 1);
    }

    fn bfs(&mut self) -> bool {
        let mut queue = VecDeque::new();
        queue.push_back(self.source);
        self.level[self.source] = 0;
        while let Some(v) = queue.pop_front() {
            for id in self.adjacency[v].iter().copied() {
                if self.edges[id].cap - self.edges[id].flow < 1 {
                    continue;
                }
                if self.level[self.edges[id].points.1] != usize::MAX {
                    continue;
                }
                self.level[self.edges[id].points.1] = self.level[v] + 1;
                queue.push_back(self.edges[id].points.1);
            }
        }
        self.level[self.sink] != usize::MAX
    }

    fn dfs(&mut self, v: usize, budget: isize) -> isize {
        stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
            if budget == 0 {
                return 0;
            }
            if v == self.sink {
                return budget;
            }
            while self.pointer[v] < self.adjacency[v].len() {
                let cid = self.pointer[v];
                self.pointer[v] += 1;
                let id = self.adjacency[v][cid];
                let u = self.edges[id].points.1;
                let space = self.edges[id].cap - self.edges[id].flow;
                if self.level[v] + 1 != self.level[u] || space < 1 {
                    continue;
                }
                let update = self.dfs(u, budget.min(space));
                if update < 1 {
                    continue;
                }
                self.edges[id].flow += update;
                self.edges[id ^ 1].flow -= update;
                return update;
            }
            0
        })
    }

    pub fn calculate_flow(&mut self) -> isize {
        while self.bfs() {
            loop {
                let update = self.dfs(self.source, isize::MAX);
                if update < 1 {
                    break;
                }
                self.flow += update;
            }
            self.reset();
        }
        self.flow
    }

    pub fn extract_current_flow(&self) -> Box<[(usize, usize, isize)]> {
        self.edges
            .iter()
            .step_by(2)
            .filter_map(|edge| {
                if edge.flow < 1 {
                    return None;
                }
                Some((edge.points.0, edge.points.1, edge.flow))
            })
            .collect()
    }

    pub fn current_flow(&self) -> impl Iterator<Item = (usize, usize, isize)> + '_ {
        self.edges.iter().step_by(2).filter_map(|edge| {
            if edge.flow < 1 {
                return None;
            }
            Some((edge.points.0, edge.points.1, edge.flow))
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn maxflow_example() {
        let mut g = DinicGraph::new(6, 0, 5);
        g.add_edge(0, 1, 16);
        g.add_edge(0, 2, 13);
        g.add_edge(1, 2, 10);
        g.add_edge(1, 3, 12);
        g.add_edge(2, 1, 4);
        g.add_edge(2, 4, 14);
        g.add_edge(3, 2, 9);
        g.add_edge(3, 5, 20);
        g.add_edge(4, 3, 7);
        g.add_edge(4, 5, 4);
        assert_eq!(g.calculate_flow(), 23);
        println!("{:?}", g.extract_current_flow());
    }
}
