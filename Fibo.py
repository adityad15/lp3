

def fibonacci_recursive(n):
    
    
    if n <= 1:
        return n
    
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_iterative(n):
    
    
    if n <= 1:
        return n

   
    prev, curr = 0, 1

    for _ in range(2, n + 1):
        
        prev, curr = curr, prev + curr

    return curr




n = int(input("Enter the position for Fibonacci calculation: "))
print("Recursive Fibonacci result:", fibonacci_recursive(n))
print("Iterative Fibonacci result:", fibonacci_iterative(n))
