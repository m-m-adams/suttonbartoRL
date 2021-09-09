use derive_more::{Index, IndexMut};
use std::cmp;

const MAX_CARS: usize = 10;

#[derive(Index, IndexMut)]
struct StateValue([[f32; MAX_CARS]; MAX_CARS]);

struct State(i32, i32);

#[derive(Index, IndexMut)]
struct StatePolicy ([[Action; MAX_CARS]; MAX_CARS]);

type Action = i32;

type Result = f32;





struct PolicyIterator {
    Values: StateValue,
    Policy: StatePolicy,
    Threshold: f32,
}



impl PolicyIterator{
    fn RentCars(&self, state: State, action: Action) -> Result {
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
    
        result += reward + 0.9 *&self.Values[rent_1st][rent_2nd];
    
        result
    }
    fn Evaluate(&mut self){
        let mut delta = 10.0;
        while delta > self.Threshold{
            for n1 in 0..(MAX_CARS+1) {
                for n2 in 0..(MAX_CARS+1){
                    let v = self.Values[n1][n2];
                    let vnext = self.RentCars(State(n1 as i32, n2 as i32), self.Policy[n1][n2]);
                    self.Values[n1][n2] = vnext;
                    delta = (v-vnext).abs()
                }
            }
        }
    }

    fn improve(&mut self){
        let mut stable = true;
        for n1 in 0..(MAX_CARS+1){
            for n2 in 0..(MAX_CARS+1){}
                aold = self.Policy[n1][n2];
                let mut v = -1000.0;
                let anext = aold;
                for a in range(-5, 6):
                    vnew = calcv((n1, n2), a, self.v)
                    #print(f"value of moving {a} from {n1} to {n2} is {vnew}")

                    if vnew > v:
                        v = vnew
                        anext = a
                if anext != aold:
                    self.p[n1, n2] = anext
                    stable = False
            }
        }
        return stable
    }
}

fn main() {
    println!("Hello, world!");
}
