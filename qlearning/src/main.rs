//use derive_more::{Index, IndexMut};
use ord_subset::OrdSubsetIterExt;

use rand::{
    distributions::{Distribution, Standard},
    random, Rng,
};
//use std::cmp;
use std::{collections::HashMap, fmt::Display};
//use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq)]
struct State(isize, isize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Action {
    Left,
    Right,
    Up,
    Down,
}

impl Distribution<Action> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Action {
        // match rng.gen_range(0, 3) { // rand 0.5, 0.6, 0.7
        match rng.gen_range(0..=3) {
            // rand 0.8
            0 => Action::Left,
            1 => Action::Right,
            2 => Action::Up,
            _ => Action::Down,
        }
    }
}

struct WindyGrid {
    pub state: State,
    xsize: isize,
    ysize: isize,
    wind: Vec<Vec<(Action, isize)>>,
    reward: Vec<Vec<isize>>,
    terminal: State,
}

impl WindyGrid {
    pub fn take_action(&mut self, a: Action) -> (State, isize, bool) {
        //let the wind move the agent
        let (w, d) = self.wind[self.state.0 as usize][self.state.1 as usize];
        self.act(w, d);

        //then let the agent take their action
        self.act(a, 1);

        let mut terminal = false;
        if self.state == self.terminal {
            terminal = true;
        }

        return (
            self.state,
            self.reward[self.state.0 as usize][self.state.1 as usize],
            terminal,
        );
    }

    fn reset(&mut self) {
        let y = rand::thread_rng().gen_range(0..self.ysize);
        let x = rand::thread_rng().gen_range(0..self.xsize);
        self.state = State(x, y);
    }

    fn act(&mut self, a: Action, d: isize) {
        match a {
            Action::Left => self.state.0 = 0.max(self.state.0 - d),
            Action::Right => self.state.0 = (self.xsize - 1).min(self.state.0 + d),
            Action::Up => self.state.1 = 0.max(self.state.1 - d),
            Action::Down => self.state.1 = (self.ysize - 1).min(self.state.1 + d),
        }
    }
}

struct Agent<'a> {
    q_table: Vec<Vec<HashMap<Action, f64>>>,
    lambda: f64,
    step_size: f64,
    epsilon: f64,
    environment: &'a mut WindyGrid,
}

impl<'a> Agent<'a> {
    fn learn(&mut self) {
        for i in 0..50000 {
            let mut r_t = 0.;
            let mut n_s = 0;
            let mut terminal = false;
            self.environment.reset();
            while !terminal {
                let state = self.environment.state;
                let action = self.select_action(state);
                let (new_state, reward, t) = self.environment.take_action(action);
                terminal = t;
                r_t += reward as f64;
                n_s += 1;
                let q_p = self.q_table(state)[&action];
                let q_n = self.q_table(new_state)[&self.optimal_q(new_state)];
                let q = q_p + self.step_size * (reward as f64 + self.lambda * q_n - q_p);
                //println!("updating key {:?} from {} to {}", action, q_p, q);
                let r = self.q_table_insert(state, action, q);
                //println!("{:?} result was {:?}", self.q_table(state), r);
                if terminal && i % 1000 == 0 {
                    println!("reward: {} in steps {}", r_t, n_s)
                }
            }
        }
    }

    fn q_table(&self, state: State) -> HashMap<Action, f64> {
        return self.q_table[state.0 as usize][state.1 as usize].clone();
    }
    fn q_table_insert(&mut self, state: State, action: Action, value: f64) -> Option<f64> {
        return self.q_table[state.0 as usize][state.1 as usize].insert(action, value);
    }

    fn select_action(&self, state: State) -> Action {
        let y: f64 = random();
        if y < self.epsilon {
            let a: Action = rand::random();
            return a;
        }

        return self.optimal_q(state);
    }

    fn optimal_q(&self, state: State) -> Action {
        return *self
            .q_table(state)
            .iter()
            .ord_subset_max_by_key(|entry| entry.1)
            .map(|(k, _v)| k)
            .unwrap();
    }
}

impl<'a> Display for Agent<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut output = String::new();
        for row in self.q_table.iter().rev() {
            for entry in row {
                let opt = entry
                    .iter()
                    .ord_subset_max_by_key(|entry| entry.1)
                    .map(|(k, _v)| k)
                    .unwrap();
                output.push_str(&format!("{:?}, ", opt));
            }
            output.push_str(&format!("\n"));
        }
        write!(f, "{}", output)
    }
}
fn main() {
    let mut default = HashMap::new();
    default.insert(Action::Down, 1.);
    default.insert(Action::Left, 0.);
    default.insert(Action::Right, 0.);
    default.insert(Action::Up, 0.);

    let gridx: isize = 10;
    let gridy: isize = 10;

    let terminal = State(8, 8);

    let mut wind = vec![vec![(Action::Left, 0isize); gridy as usize]; gridx as usize];

    wind[6] = vec![(Action::Up, 2); gridy as usize];
    wind[5] = vec![(Action::Up, 1); gridy as usize];
    wind[7] = vec![(Action::Up, 1); gridy as usize];

    let mut reward = vec![vec![-1; gridy as usize]; gridx as usize];
    reward[terminal.0 as usize][terminal.1 as usize] = 1;

    let mut env = WindyGrid {
        state: State(0, 0),
        xsize: gridx,
        ysize: gridy,
        wind: wind,
        reward: reward,
        terminal: terminal,
    };

    let mut agent = Agent {
        q_table: vec![vec![default.clone(); gridy as usize]; gridx as usize],
        lambda: 0.9,
        step_size: 0.01,
        epsilon: 0.01,
        environment: &mut env,
    };
    println!("{}", agent);
    agent.learn();
    println!("{}", agent)
}
