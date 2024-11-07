

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


def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def fibonacci_iterative_sequence(n):
    sequence = []
    prev, curr = 0, 1
    for i in range(n):
        sequence.append(prev)
        prev, curr = curr, prev + curr
    return sequence

n = int(input("Enter the number of terms for the Fibonacci sequence: "))

print("Iterative Fibonacci sequence:", fibonacci_iterative_sequence(n))
print("Recursive Fibonacci sequence:", [fibonacci_recursive(i) for i in range(n)])