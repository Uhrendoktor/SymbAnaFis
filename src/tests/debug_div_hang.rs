#[cfg(test)]
mod tests {
    use crate::diff;

    #[test]
    fn test_div_hang_repro() {
        // sin(x)/x
        println!("Starting differentiation of sin(x)/x");
        let result = diff("sin(x)/x", "x", &[], None);
        match result {
            Ok(res) => println!("Result: {}", res),
            Err(e) => println!("Error: {:?}", e),
        }
    }
}
