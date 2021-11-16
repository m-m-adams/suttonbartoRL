use rand::{
    distributions::{Distribution, Standard},
    random, Rng,
};
type State = isize;
type Terminal = bool;
type Reward = f64;
type RewardState = (Reward, State, Terminal);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Action {
    Left,
    Right,
}

impl Distribution<Action> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Action {
        match rng.gen_range(0..=1) {
            0 => Action::Left,
            _ => Action::Right,
        }
    }
}

struct RandomWalkEnvironment {
    position: State,
    start: State,
    left_terminal: State,
    right_terminal: State,
    num_states: isize,
}

impl RandomWalkEnvironment {
    pub fn init(&mut self) -> RewardState {
        self.position = self.start;
        return self.reward();
    }

    pub fn step(&mut self, dir: Action) -> RewardState {
        let p = self.position;
        let dis: isize = rand::thread_rng().gen_range(0..100);
        match dir {
            Action::Left => self.position = self.left_terminal.max(p - dis),
            Action::Right => self.position = self.right_terminal.min(p + dis),
        }

        return self.reward();
    }

    fn reward(&self) -> RewardState {
        match self.position {
            p if p == self.left_terminal => return (-1., p, true),
            p if p == self.right_terminal => return (1., p, true),
            p => return (0., p, false),
        }
    }
}

struct Agent {
    num_states: isize,
    num_groups: isize,
    step_size: f64,
    discount: f64,
    last_state: Option<State>,
    last_action: Option<Action>,
    //state_vector: Vec<isize>,
    last_bin: usize,
    weights: Vec<f64>,
}

fn build_agent(num_states: isize, num_groups: isize, step_size: f64, discount: f64) -> Agent {
    Agent {
        num_states,
        num_groups,
        step_size,
        discount,
        last_action: None,
        last_state: None,
        last_bin: 0,

        weights: vec![0.; num_groups as usize],
    }
}

impl Agent {
    fn policy(&self, _: State) -> Action {
        let d: Action = rand::thread_rng().gen();
        d
    }

    fn to_bin(&mut self, s: State) -> usize {
        let states_per_group = self.num_states / self.num_groups;
        let bin = s / states_per_group;

        //self.last_bin = bin;
        bin.min(self.num_groups - 1) as usize
    }

    pub fn start(&mut self, state: State) -> Action {
        self.last_state = Some(state);
        let a = self.policy(state);
        self.last_action = Some(a);
        return a;
    }

    pub fn step(&mut self, srt: RewardState) -> Action {
        let (reward, state, _) = srt;

        let bin_c = self.to_bin(state);

        //inc using the td(0) update with state aggregation for weights
        self.weights[self.last_bin] += self.step_size
            * (reward + self.discount * self.weights[bin_c] - self.weights[self.last_bin]);
        self.last_bin = bin_c;
        let a = self.policy(state);
        self.last_action = Some(a);
        return a;
    }

    pub fn finish(&mut self, srt: RewardState) {
        let (reward, state, _) = srt;

        self.weights[self.last_bin] += self.step_size * (reward - self.weights[self.last_bin]);
    }
}

fn main() {
    let mut env = RandomWalkEnvironment {
        position: 250,
        start: 250,
        left_terminal: 0,
        right_terminal: 500,
        num_states: 500,
    };

    let mut ag = build_agent(500, 10, 0.1, 0.9);
    for trial in 0..100 {
        let mut terminal = false;
        env.init();
        let a = ag.start(env.start);
        while !terminal {
            let srt = env.step(a);

            let a = ag.step(srt);
            terminal = srt.2;
        }
        println!("{:?}", ag.weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_agent_end() {
        let mut ag = build_agent(500, 10, 0.1, 0.9);
        ag.weights = [-1.5, 0.5, 1., -0.5, 1.5, -0.5, 1.5, 0.0, -0.5, -1.0].to_vec();
        let start_state = 50;
        let _ = ag.start(start_state);
        ag.finish((10., 1, true));
        println!("{:?}", ag.weights);
        assert!(-0.35 - ag.weights[0] < 0.001);
    }

    #[test]
    fn test_agent_step() {
        let mut ag = build_agent(500, 10, 0.1, 0.9);
        ag.weights = [-1.5, 0.5, 1., -0.5, 1.5, -0.5, 1.5, 0.0, -0.5, -1.0].to_vec();
        let start_state = 50;
        let _ = ag.start(start_state);
        ag.step((10., 120, false));
        println!("{:?}", ag.weights);
        assert!(-0.26 - ag.weights[0] < 0.001);
    }
}
