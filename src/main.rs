use derive_more::{Index, IndexMut};
use std::cmp;
use std::fmt;
const MAX_CARS: usize = 20;
const RET_1ST: f64 = 2.;
const RET_2ND: f64 = 4.;

#[derive(Index, IndexMut, Debug)]
struct StateValue([[f64; MAX_CARS + 1]; MAX_CARS + 1]);

struct State(i32, i32);

#[derive(Index, IndexMut, Debug)]
struct StatePolicy([[Action; MAX_CARS + 1]; MAX_CARS + 1]);

type Action = i32;

type Result = f64;

fn factorial(num: u64) -> u64 {
    match num {
        0 => 1,
        1 => 1,
        _ => factorial(num - 1) * num,
    }
}

fn poisson(k: f64, lambda: f64) -> f64 {
    let ret = lambda.powf(k) * (-lambda).exp() / factorial(k as u64) as f64;
    //println!("p of {} is {}", k, ret);
    ret
}

struct PolicyIterator {
    values: StateValue,
    policy: StatePolicy,
    threshold: f64,
}

impl PolicyIterator {
    fn rent_cars(&self, state: State, action: Action) -> Result {
        if (action > state.0) || (-state.1 > action) {
            return -1000.0;
        }

        let mut result = 0f64;
        //cost of moving each car
        result -= 0.0 * (action as f64).abs();

        //set the number of cars in each lot after move
        let mut n1st: i32 = cmp::min(state.0 - action, MAX_CARS as i32);
        let mut n2nd: i32 = cmp::min(state.1 + action, MAX_CARS as i32);

        for req1 in 0..10 {
            for req2 in 0..10 {
                let p = poisson(req1 as f64, RET_1ST) * poisson(req2 as f64, RET_2ND);
                //let p = 0.01;
                //let p = 1.;
                //check number of valid rentals (rentals where a car is available)
                let valid1st = cmp::min(n1st, req1);
                let valid2nd = cmp::min(n2nd, req2);
                //add reward per rental
                let reward = 10.0 * (valid1st + valid2nd) as f64;

                //remove rented cars
                n1st -= valid1st;
                n2nd -= valid2nd;

                //return cars
                let rent_1st = cmp::min((n1st as usize) + 1, MAX_CARS);
                let rent_2nd = cmp::min((n2nd as usize) + 0, MAX_CARS);

                //add reward for renting
                result += p * (reward + self.values[rent_1st][rent_2nd]);
                //println!("result is {}", reward)
            }
        }

        result
    }
    fn evaluate(&mut self) {
        let mut delta = 10.0;
        while delta > self.threshold {
            for n1 in 0..(MAX_CARS + 1) {
                for n2 in 0..(MAX_CARS + 1) {
                    let v = self.values[n1][n2].clone();

                    let vnext = self.rent_cars(State(n1 as i32, n2 as i32), self.policy[n1][n2]);

                    self.values[n1][n2] = vnext;
                    delta = (v - vnext).abs()
                }
            }
        }
    }

    fn improve(&mut self) -> bool {
        let mut stable = true;
        for n1 in 0..(MAX_CARS + 1) {
            for n2 in 0..(MAX_CARS + 1) {
                let aold = self.policy[n1][n2].clone();
                let mut v = -1000.0;
                let mut anext = aold;
                for a in -5..5 {
                    let vnew = self.rent_cars(State(n1 as i32, n2 as i32), a);

                    //println!("action {} in state {},{} had value {}", a, n1, n2, vnew);

                    if vnew > v {
                        v = vnew;
                        anext = a;
                    }
                }
                if anext != aold {
                    self.policy[n1][n2] = anext;
                    //println!("policy of {},{} changed from {} to {}", n1, n2, aold, anext);

                    stable = false;
                }
            }
        }

        return stable;
    }

    fn run(&mut self) {
        println!("{:?}", self.policy);
        self.evaluate();
        println!("{:?}", self.values);
        while !self.improve() {
            println!("{:?}", self.policy);
            self.evaluate();
            println!("{:?}", self.values);
        }
        println!("{:?}", self.policy);
    }
}

fn main() {
    println!("Hello, world!");

    let mut piter = PolicyIterator {
        values: StateValue([[0f64; MAX_CARS + 1]; MAX_CARS + 1]),
        policy: StatePolicy([[0i32; MAX_CARS + 1]; MAX_CARS + 1]),
        threshold: 1f64,
    };

    piter.run();
}
