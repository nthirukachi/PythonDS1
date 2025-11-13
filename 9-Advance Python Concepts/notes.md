# Iterators

## 1. Where to use
- When you need to process elements one by one, such as looping through a list, tuple, or any collection.
- When you want to create your own objects that can be looped over using a `for` loop.

## 2. How to use
- Most built-in collections in Python (like lists, tuples, dictionaries) are iterable, meaning you can loop through them.
- To make your own object iterable, you need to define two special methods:
  - `__iter__()` returns the iterator object itself.
  - `__next__()` returns the next value and raises `StopIteration` when done.

## 3. Why to use
- Iterators allow you to process data one item at a time, which is memory efficient, especially for large datasets.
- They provide a standard way to loop through data.

## 4. Different ways to use with examples and output

### Example 1: Using built-in iterators

```python
nums = [1, 2, 3]
it = iter(nums)  # Get an iterator from the list
print(next(it))  # Output: 1
print(next(it))  # Output: 2
print(next(it))  # Output: 3
# print(next(it))  # This would raise StopIteration
```
**Output:**
```
1
2
3
```
- `iter(nums)` creates an iterator object.
- `next(it)` gets the next item each time.

### Example 2: Custom iterator

```python
class Counter:
    def __init__(self, low, high):
        self.current = low
        self.high = high
    def __iter__(self):
        return self
    def __next__(self):
        if self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            return self.current - 1

for num in Counter(1, 3):
    print(num)
```
**Output:**
```
1
2
3
```
- This class counts from `low` to `high`.
- The `for` loop automatically uses the iterator protocol.

## 5. Most commonly used with example

- Reading files line by line:
```python
for line in open('file.txt'):
    print(line.strip())
```
**Output:** (Assuming file.txt contains three lines: hello, world, python)
```
hello
world
python
```
- This reads one line at a time, which is memory efficient.

## 6. Best way / Real world use case

- Iterators are best when you want to process large data sets without loading everything into memory, such as reading big files or streaming data.

## Cheat codes
- Use `iter()` to get an iterator.
- Use `next()` to get the next item.
- Use `for` loops for easy iteration.

---

# Generators

## 1. Where to use
- When you want to generate a sequence of values, especially if the sequence is large or even infinite.
- When you want to write functions that can pause and resume their execution.

## 2. How to use
- Use the `yield` keyword in a function to turn it into a generator.
- Each time you call `next()` on the generator, it resumes where it left off.

## 3. Why to use
- Generators are memory efficient because they yield items one at a time.
- They are useful for pipelines and streaming data.

## 4. Different ways to use with examples and output

### Example 1: Generator function

```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for i in countdown(3):
    print(i)
```
**Output:**
```
3
2
1
```
- The function `countdown` yields numbers from `n` down to 1.
- Each time through the loop, the function resumes after the last `yield`.

### Example 2: Generator expression

```python
squares = (x*x for x in range(5))
for sq in squares:
    print(sq)
```
**Output:**
```
0
1
4
9
16
```
- This is like a list comprehension, but with parentheses.
- It generates each square one at a time.

## 5. Most commonly used with example

- Reading large files lazily:
```python
def read_large_file(file_name):
    with open(file_name) as f:
        for line in f:
            yield line

for line in read_large_file('file.txt'):
    print(line.strip())
```
**Output:** (Assuming file.txt contains three lines: hello, world, python)
```
hello
world
python
```

## 6. Best way / Real world use case

- Generators are best for pipelines, streaming data, or when you want to process data as it comes in (like reading logs or processing sensor data).

## Cheat codes
- Use `yield` in a function to make a generator.
- Use generator expressions: `(x for x in y)`.

---

# Decorators

## 1. Where to use
- When you want to add extra functionality to a function or method without changing its code.
- Common uses: logging, timing, access control, caching, etc.

## 2. How to use
- Write a function that takes another function as input and returns a new function.
- Use the `@decorator_name` syntax above the function you want to decorate.

## 3. Why to use
- Decorators help you reuse code and keep your functions clean.
- They separate concerns (e.g., logging vs. business logic).

## 4. Different ways to use with examples and output

### Example 1: Simple decorator

```python
def my_decorator(func):
    def wrapper():
        print("Before")
        func()
        print("After")
    return wrapper

@my_decorator
def say_hello():
    print("Hello")

say_hello()
```
**Output:**
```
Before
Hello
After
```
- The decorator adds behavior before and after the original function.

### Example 2: Decorator with arguments

```python
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def greet():
    print("Hi!")

greet()
```
**Output:**
```
Hi!
Hi!
Hi!
```
- This decorator repeats the function call `n` times.

## 5. Most commonly used with example

- Timing how long a function takes:
```python
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Elapsed: {time.time() - start:.2f} seconds")
        return result
    return wrapper

@timer
def slow_func():
    time.sleep(1)

slow_func()
```
**Output:**
```
Elapsed: 1.00 seconds
```
- The decorator measures and prints how long the function takes to run.

## 6. Best way / Real world use case

- Decorators are widely used in web frameworks (like Flask or Django) for routes, authentication, and permissions.
- Example: In Flask,
```python
@app.route('/home')
def home():
    return "Welcome!"
```
- Here, `@app.route` is a decorator that registers the function as a web route.

## Cheat codes
- Use `functools.wraps` to preserve the original function's metadata.
- You can stack multiple decorators on a single function.

---

# Comparison

| Feature      | Iterators                | Generators                | Decorators                      |
|--------------|--------------------------|---------------------------|----------------------------------|
| Purpose      | Traverse collections     | Generate sequences lazily | Modify/enhance functions/methods |
| Syntax       | `__iter__`, `__next__`   | `yield` in functions      | `@decorator` above functions     |
| Memory       | Efficient (one at a time)| Very efficient            | N/A                              |
| Use case     | Custom loops, files      | Pipelines, streams        | Logging, timing, auth            |
| Example      | `for x in obj:`          | `def gen(): yield ...`    | `@my_decorator`                  |

---

**Summary:**
- **Iterators** help you loop through data one item at a time.
- **Generators** are a special type of iterator, created with functions and `yield`, for generating data on the fly.
- **Decorators** let you add new features to functions easily, without changing their code.

If you practice these examples and see their outputs, you'll understand how powerful and useful these concepts are in Python!

