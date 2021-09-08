use derive_more::{Index};
use std::cmp;

const MAX_CARS: usize = 10;

#[derive(Index)]
struct StateValue([[f32; MAX_CARS]; MAX_CARS]);

struct State(i32, i32);

#[derive(Index)]
struct StatePolicy ([[Action; MAX_CARS]; MAX_CARS]);

type Action = i32;

type Result = f32;



fn RentCars(state: State, action: Action, state_value: StateValue) -> Result {
    if (action > state.0) || (-state.1 > action) {
        return -1000.0
    }

    let mut result = 0.0;

    result -= 2.0*(action as f32).abs();
    
    let mut n1st: i32 = cmp::min(state.0 - action, MAX_CARS as i32);
    let mut n2nd: i32 = cmp::min(state.1 + action, MAX_CARS as i32);

    let valid1st = cmp::min(n1st, 3);
    let valid2nd = cmp::min(n2nd, 4);
    let reward = 10.0*(valid1st+valid2nd) as f32;
    n1st -= valid1st;
    n2nd -= valid2nd;

    let rent_1st = cmp::min((n1st as usize)+3, MAX_CARS);
    let rent_2nd = cmp::min((n2nd as usize)+2, MAX_CARS);

    result += reward + 0.9 *state_value[rent_1st][rent_2nd];

    result
}

struct PolicyIterator {
    Values: StateValue,
    Policy: StatePolicy,
    Threshold: f32,
}

impl PolicyIterator{

}

fn main() {
    println!("Hello, world!");
}
