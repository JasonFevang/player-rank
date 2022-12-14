use ndarray::*;
use ndarray_linalg::*;
use rand::Rng;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::io;

const NUM_PLAYERS: usize = 4;

// Generate a vector of randomized rankings to seed the matrix with instead of asking questions
// Results are normalized to the last entry, so the last entry will always be 1
fn gen_seed_rankings() -> Vec<f64>{
    let mut seed_rankings: Vec<f64> = Vec::new();
    for _ in 0..NUM_PLAYERS{
        seed_rankings.push(rand::random::<f64>());
    }

    // Normalize seed rankings
    for i in 0..NUM_PLAYERS{
        seed_rankings[i] = seed_rankings[i] / seed_rankings[NUM_PLAYERS - 1];
    }
    seed_rankings
}

// Given a seed vector of rankings of players, generate a test linear system
fn gen_lin_sys_from_seed_rankings(seed_rankings: &Vec<f64>, noise_range: f64, chance_skip_row: f64) -> (Array2<f64>, Array1<f64>){
    // Generate A matrix
    let mut a: Array2<f64> = arr2(&[[]]);
    // First row
    for _ in 0..NUM_PLAYERS{
        a.push_column(ArrayView::from(&[0.])).unwrap();
    }
    a[[0, NUM_PLAYERS - 1]] = 1.;

    // Remaining rows
    for p1 in 0..NUM_PLAYERS{
        for p2 in (p1 + 1)..NUM_PLAYERS{
            // Skip rows occasionally
            if rand::random::<f64>() < chance_skip_row{
                continue;
            }

            let mut next_row: [f64; NUM_PLAYERS] = [0.; NUM_PLAYERS];
            // Generate random noise on the accuracy of the rankings
            let noise: f64 = rand::thread_rng().gen_range((1. - noise_range/2.)..(1. + noise_range/2.));
            next_row[p1] = 1.;
            next_row[p2] = -seed_rankings[p1]/seed_rankings[p2] * noise; // noisy seed ranking
            a.push_row(ArrayView::from(&next_row)).unwrap();
        }
    }

    // Generate b vector
    let mut b: Vec<f64> = vec![0.; a.dim().0];
    b[0] = 1.;
    let b: Array1<f64> = arr1(&b);

    (a, b)
}

enum UserResponse{
    Value(f64),
    Skip,
    Quit
}

fn get_response() -> UserResponse{
    loop {
        let mut input = String::new();

        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");
        let input = input.trim();

        if input == "s" {
            return UserResponse::Skip;
        }
        if input == "q" {
            return UserResponse::Quit;
        }

        if let Ok(val) = input.parse::<f64>(){
            return UserResponse::Value(val);
        };
    }
}

fn ask_self_question(name: &str) -> UserResponse{
    println!("{} Attack v {} Defense", name, name);
    return get_response();
}

// Ask user for an input on how much better name1 is than name2
fn ask_question(name1: &str, name2: &str) -> UserResponse{
    println!("{} v {}", name1, name2);
    return get_response();
}

// Generate a randomized, minimum set of questions to fully define the vector space(idk if that means anything but it sounds sick lmao)
fn gen_min_questions(num_players: usize) -> Vec<(usize, usize)>{
    // Create shuffled list of all players
    let mut player_list: Vec<usize> = (0..num_players).collect();
    let mut rng = thread_rng();
    player_list.shuffle(&mut rng);

    // Create pairs from this shuffled list
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..(num_players - 1){
        pairs.push((player_list[i], player_list[i + 1]));
    }

    // Shuffle those pairs
    pairs.shuffle(&mut rng);
    pairs
}

// When a pair gets skipped, find a pair to replace it while maintaining the requirements of a fully connected graph
fn replace_skipped_pair(upcoming: &Vec<(usize, usize)>, skipped: &Vec<(usize, usize)>, answered: &Vec<(usize, usize)>, pair: &(usize, usize)) -> Option<(usize, usize)>{
    // Find all numbers connected to each number in the skipped pair
    let mut lhs: Vec<usize> = vec![pair.0];
    let mut rhs: Vec<usize> = vec![pair.1];

    let mut all_pairs: Vec<(usize, usize)> = upcoming.clone();
    all_pairs.extend(answered);

    // Iterate through all the pairs
    // If one of them connects to a known number, add it to the list and start looking for another
    while !all_pairs.is_empty(){
        for i in (0..all_pairs.len()).rev() {
            let pair = all_pairs[i];
            let mut found = false;
            if lhs.contains(&pair.0){
                lhs.push(pair.1);
                found = true;
            }
            else if lhs.contains(&pair.1){
                lhs.push(pair.0);
                found = true;
            }
            else if rhs.contains(&pair.0){
                rhs.push(pair.1);
                found = true;
            }
            else if rhs.contains(&pair.1){
                rhs.push(pair.0);
                found = true;
            }
            if found{
                all_pairs.remove(i);
                break;
            }
        }
    }

    // Generate list of all potential replacement pairs, not including the skipped ones
    let mut potential_replacements: Vec<(usize, usize)> = Vec::new();
    for left in &lhs{
        for right in &rhs{
            let potential_pair = (*left, *right);
            let potential_pair_rev = (*right, *left);
            if !skipped.contains(&potential_pair) && !skipped.contains(&potential_pair_rev){
                potential_replacements.push(potential_pair);
            }
        }
    }

    // If all the options are skipped, return an error
    if potential_replacements.is_empty(){
        return None;
    }

    // Choose a random one and add it to the upcoming list
    let rand = rand::random::<usize>() % potential_replacements.len();
    Some(potential_replacements[rand])
}


fn count_connections(answered: &Vec<(usize, usize)>, pair: (usize, usize)) -> usize{
    let mut left_cnt = 0;
    let mut right_cnt = 0;
    for p in answered{
        if p.0 == pair.0 || p.1 == pair.0{
            left_cnt += 1;
        }
        if p.0 == pair.1 || p.1 == pair.1{
            right_cnt += 1;
        }
    }
    left_cnt + right_cnt
}

fn ask_self_questions(names: &Vec<String>) -> Vec<(usize, f64)>{
    let mut results: Vec<(usize, f64)> = Vec::new();

    // Create shuffled list of all players
    let mut player_list: Vec<usize> = (0..names.len()).collect();
    player_list.shuffle(&mut thread_rng());

    for player in player_list{
        let response = ask_self_question(&names[player]);
        if let UserResponse::Value(val) = response {
            results.push((player, val));
        }
        else if let UserResponse::Skip = response {
            continue;
        }
        else if let UserResponse::Quit = response {
            break;
        }
    }

    results
}
// Given a list of names, ask a series of questions comparing two players
// The user can provide a value, skip a question, or quit whenever
// Returns a list of pair of players(as indices) and the rating between them
fn ask_questions(names: &Vec<String>) -> Vec<(usize, usize, f64)>{
    let mut results: Vec<(usize, usize, f64)> = Vec::new();
    let mut upcoming = gen_min_questions(names.len());
    let mut skipped: Vec<(usize, usize)> = Vec::new();
    // This is a duplicate of results. I don't care
    let mut answered: Vec<(usize, usize)> = Vec::new();

    // Ask necessary questions
    while !upcoming.is_empty(){
        let pair = upcoming.pop().unwrap();
        let (p1, p2) = pair;

        // Ask a question, get a response or break
        let response = ask_question(&names[p1], &names[p2]);
        if let UserResponse::Value(val) = response {
            results.push((p1, p2, val));
            answered.push(pair);
        }
        else if let UserResponse::Skip = response {
            // Perform skip logic. Append replacement questions to the min question set
            skipped.push(pair);
            match replace_skipped_pair(&upcoming, &skipped, &answered, &pair){
                Some(val) => upcoming.push(val),
                None => assert!(false) // Skipped too many, need to reanswer some of the skipped ones
            };
        }
        else if let UserResponse::Quit = response {
            println!("Breaking early, this isn't going to work");
            assert!(false);
        }
    }

    println!("Done necessary questions");

    let mut prev_min_links = 0;
    loop {
        // List all remaining questions
        let mut remaining_questions: Vec<(usize, usize)> = Vec::new();
        for p1 in 0..names.len(){
            for p2 in (p1 + 1)..names.len(){
                let pair = (p1, p2);
                let pair_rev = (p2, p1);
                if !answered.contains(&pair) && !answered.contains(&pair_rev) && 
                    !skipped.contains(&pair) && !skipped.contains(&pair_rev){
                    remaining_questions.push((p1, p2));
                }
            }
        }
        remaining_questions.shuffle(&mut thread_rng());

        // println!("Remaining: {:?}", remaining_questions);

        // Find minimum linked question in the list
        let mut min_links = names.len();
        let mut min_pair: Option<&(usize, usize)> = None;
        for pair in &remaining_questions{
            let pair_links = count_connections(&answered, *pair) / 2;
            if pair_links < min_links {
                min_links = pair_links;
                min_pair = Some(pair);
            }
        }

        // Update user on how connected the graph is
        // This is sorta like a confidence measure
        if min_pair.is_none(){
            println!("Fully linked. You answered or skipped every question");
            break;
        }
        if prev_min_links != min_links{
            if min_links >= 5{
                println!("{}-linked", min_links);
            }
            else if min_links >= 4{
                println!("Quadruply linked");
            }
            else if min_links >= 3{
                println!("Triply linked");
            }
            else if min_links >= 2{
                println!("Doubly linked");
            }
            prev_min_links = min_links;
        }

        // if min_pair.is_some(){
        //     println!("Min pair is {:?} with count {}", min_pair, min_links);
        // }

        let min_pair = min_pair.unwrap();
        let (p1, p2) = *min_pair;

        // Ask the question
        let response = ask_question(&names[p1], &names[p2]);
        if let UserResponse::Value(val) = response{
            answered.push(*min_pair);
            results.push((p1, p2, val));
        }
        else if let UserResponse::Skip = response {
            skipped.push(*min_pair);
        }
        else if let UserResponse::Quit = response {
            break;
        }
    }

    results
}

fn gen_lin_sys_from_goalie_results(goalie_results: Vec<(usize, usize, f64)>, num_players: usize) -> (Array2<f64>, Array1<f64>){
    // Generate A matrix
    let mut a: Array2<f64> = arr2(&[[]]);
    // First row
    for _ in 0..num_players{
        a.push_column(ArrayView::from(&[0.])).unwrap();
    }
    a[[0, num_players - 1]] = 1.;

    // Add self results
    for entry in goalie_results{
        let mut next_row: Vec<f64> = vec![0.; num_players];
        next_row[entry.0] = 1.;
        next_row[entry.1] = -entry.2;
        a.push_row(ArrayView::from(&next_row)).unwrap();
    }

    // Generate b vector
    let mut b: Vec<f64> = vec![0.; a.dim().0];
    b[0] = 1.;
    let b: Array1<f64> = arr1(&b);

    (a, b)
}

// Convert a list of player-pairs and relative ratings, generate a system 
// of equations and solve
fn gen_lin_sys_from_outfield_results(attack_results: Vec<(usize, usize, f64)>, defense_results: Vec<(usize, usize, f64)>, self_results: Vec<(usize, f64)>, num_players: usize) -> (Array2<f64>, Array1<f64>){
    // Generate A matrix
    let mut a: Array2<f64> = arr2(&[[]]);
    // First row
    for _ in 0..num_players*2{
        a.push_column(ArrayView::from(&[0.])).unwrap();
    }
    a[[0, num_players - 1]] = 1.;

    // Add attack results
    for entry in attack_results{
        let mut next_row: Vec<f64> = vec![0.; num_players*2];
        next_row[entry.0] = 1.;
        next_row[entry.1] = -entry.2;
        a.push_row(ArrayView::from(&next_row)).unwrap();
    }

    // Add defense results
    for entry in defense_results{
        let mut next_row: Vec<f64> = vec![0.; num_players*2];
        next_row[num_players+entry.0] = 1.;
        next_row[num_players+entry.1] = -entry.2;
        a.push_row(ArrayView::from(&next_row)).unwrap();
    }

    // Add self results
    for entry in self_results{
        let mut next_row: Vec<f64> = vec![0.; num_players*2];
        next_row[entry.0] = 1.;
        next_row[num_players+entry.0] = -entry.1;
        a.push_row(ArrayView::from(&next_row)).unwrap();
    }

    // Generate b vector
    let mut b: Vec<f64> = vec![0.; a.dim().0];
    b[0] = 1.;
    let b: Array1<f64> = arr1(&b);

    (a, b)
}

// Solve a linear system using a least squares regression
fn least_squares_regression(a: Array2<f64>, b: Array1<f64>) -> Vec<f64>{
    // Perform linear algebra to do least squares regression
    let ata = a.t().dot(&a);
    let atb = a.t().dot(&b);
    let mut sol = ata.solve(&atb).unwrap().to_vec();

    // Normalize solution vector to the last element
    for i in 0..NUM_PLAYERS{
        sol[i] = sol[i] / sol[NUM_PLAYERS - 1];
    }
    sol
}

// Given the seed ranks and the computed result ranks, compute a vector or %error on the computation
fn compute_err_from_seed(seed_rankings: &Vec<f64>, sol: &Vec<f64>) -> Vec<f64>{
    let mut err: Vec<f64> = Vec::new();
    for i in 0..NUM_PLAYERS{
        err.push(((seed_rankings[i] - sol[i])/seed_rankings[i]).abs());
    }
    err
}

// Compute and log max and mean errors on all the players
fn log_err_stats(err: &Vec<f64>){
    let mut max_err = err[0];
    let mut sum = 0.;
    for er in err{
        sum += er;
        if *er > max_err {
            max_err = *er;
        }
    }
    let mean = sum / err.len() as f64;
    println!("Max: {:.3} Mean: {:.3}", max_err, mean);
}

fn names_from_file(filename: &str) -> (Vec<String>, Vec<String>, Vec<Option<usize>>){
    let mut names: Vec<String> = Vec::new();
    let mut goalies: Vec<String> = Vec::new();
    let mut goalies_map: Vec<Option<usize>> = Vec::new();
    let mut rdr = csv::Reader::from_path(filename).unwrap();
    for result in rdr.records() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here.
        let record = result.unwrap();
        if let Some(name) = record.get(0){
            names.push(String::from(name));
            let mut goalies_map_val: Option<usize> = None;
            if let Some(goalie_field) = record.get(1){
                if goalie_field == "yes"{
                    goalies.push(String::from(name));
                    goalies_map_val = Some(goalies.len() - 1);
                }
            }
            goalies_map.push(goalies_map_val);
        }
    }
    (names, goalies, goalies_map)
}

fn write_results_to_file(filename: &str, results: &Vec<(&str, f64, f64, f64)>){
    let mut wtr = csv::Writer::from_path(filename).unwrap();
    wtr.write_record(&["Name", "Atk", "Def", "Goal"]).unwrap();
    for row in results{
        wtr.write_record(&[row.0, &format!("{:.2}", row.1), &format!("{:.2}", row.2), &format!("{:.2}", row.3)]).unwrap();
    }
    wtr.flush().unwrap();
}

fn main() {
    let auto_test = false;
    let read_from_csv = true;
    let names_filename = "test.csv";
    let output_filename = "results.csv";
    if auto_test{
        let seed_rankings = gen_seed_rankings();
        let noise_range = 0.2;
        let chance_skip_row = 0.5;
        let (a, b) = gen_lin_sys_from_seed_rankings(&seed_rankings, noise_range, chance_skip_row);
        let sol = least_squares_regression(a, b);
        let err = compute_err_from_seed(&seed_rankings, &sol);
        log_err_stats(&err);
    }
    else{
        let (names, goalies, goalies_map) = if read_from_csv{
            names_from_file(names_filename)
        }
        else{
            (vec![
                String::from("P1"),
                String::from("P2"),
                String::from("P3"),
                String::from("P4"),
                String::from("P5"),
                String::from("P6"),
            ], vec![
                String::from("P1"),
                String::from("P2"),
                String::from("P3"),
                String::from("P4"),
                String::from("P5"),
                String::from("P6"),
            ], vec![
                Some(0), Some(1), Some(2), Some(3), Some(4), Some(5)
            ])
        };
        println!("Attack Questions");
        let attack_results = ask_questions(&names);
        println!("Defensive Questions");
        let defense_results = ask_questions(&names);
        println!("Self Compare Questions");
        let player_comp_results = ask_self_questions(&names);
        println!("Goalie Questions");
        let goalie_results = ask_questions(&goalies);

        // Generate matrices to model the scenario and solve them
        let (a, b) = gen_lin_sys_from_outfield_results(attack_results, defense_results, player_comp_results, names.len());
        let outfield_sol = least_squares_regression(a, b);
        let (a, b) = gen_lin_sys_from_goalie_results(goalie_results, goalies.len());
        let goalie_sol = least_squares_regression(a, b);

        // Compiles linear equation solution vectors into a results vector
        let mut results: Vec<(&str, f64, f64, f64)> = Vec::new();
        for i in 0..names.len(){
            // If this player can be a goalie, load it's computed goalie score using the goalies_map to map between player index and goalie index
            // Default to 0 otherwise
            let goalie_score = match goalies_map[i]{
                Some(val) => goalie_sol[val],
                None => 0.0
            };
            results.push((&names[i], outfield_sol[i], outfield_sol[names.len() + i], goalie_score));
        }
        // Sort results by attack rating by default
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        write_results_to_file(output_filename, &results);
    }
}

