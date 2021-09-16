use derive_more::{Index, IndexMut};
use std::cmp;
use std::collections::HashMap;
use std::fmt;
use std::thread::AccessError;

#[derive(Clone, Copy)]
struct State(isize, isize);

#[derive(Clone, Copy)]
enum Action {
    left,
    right,
    up,
    down,
}

struct WindyGrid {
    state: State,
    xsize: isize,
    ysize: isize,
    wind: Vec<Vec<(Action, isize)>>,
    reward: Vec<Vec<isize>>,
    terminal: State,
}

impl WindyGrid {
    pub fn take_action(&mut self, a: Action) -> (State, isize) {
        //let the wind move the agent
        let (w, d) = self.wind[self.state.0 as usize][self.state.1 as usize];
        self.act(w, d);

        //then let the agent take their action
        self.act(a, 1);

        return (
            self.state,
            self.reward[self.state.0 as usize][self.state.1 as usize],
        );
    }

    fn act(&mut self, a: Action, d: isize) {
        match a {
            Action::left => self.state.0 = 0.max(self.state.0 - d),
            Action::right => self.state.0 = self.xsize.min(self.state.0 + d),
            Action::up => self.state.1 = 0.max(self.state.1 - d),
            Action::down => self.state.1 = self.ysize.min(self.state.1 + d),
        }
    }
}

struct Agent {
    q_table: Vec<Vec<HashMap<Action, f64>>>,
    policy: Vec<Vec<Action>>,
    lambda: f64,
}

impl Agent {
    fn learn(&mut self) {
        let n_rounds = 0;
    }
}

impl Agent {}
fn main() {
    println!("Hello, world!");
}
