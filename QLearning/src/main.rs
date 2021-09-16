//use derive_more::{Index, IndexMut};
use ord_subset::OrdSubsetIterExt;

use rand::{
    distributions::{Distribution, Standard},
    random, Rng,
};
//use std::cmp;
use std::collections::HashMap;
//use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq)]
struct State(isize, isize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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

    fn act(&mut self, a: Action, d: isize) {
        match a {
            Action::Left => self.state.0 = 0.max(self.state.0 - d),
            Action::Right => self.state.0 = self.xsize.min(self.state.0 + d),
            Action::Up => self.state.1 = 0.max(self.state.1 - d),
            Action::Down => self.state.1 = self.ysize.min(self.state.1 + d),
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
        let mut terminal = false;
        for _ in 0..1000 {
            while !terminal {
                let state = self.environment.state;
                let action = self.select_action(state);
                let (new_state, reward, terminal) = self.environment.take_action(action);
                let q_p = self.q_table(state)[&action];
                let q_n = self.q_table(new_state)[&self.optimal_q(new_state)];
                self.q_table(state).insert(
                    action,
                    q_p + self.step_size * (reward as f64 + self.lambda * q_n - q_p),
                );

                if terminal {
                    println!("{}", reward)
                }
            }
        }
    }

    fn q_table(&self, state: State) -> HashMap<Action, f64> {
        return self.q_table[state.0 as usize][state.1 as usize].clone();
    }

    fn select_action(&self, state: State) -> Action {
        let y: f64 = random();
        if y > self.epsilon {
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

fn main() {}
