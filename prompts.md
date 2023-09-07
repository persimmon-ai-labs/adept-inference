# Prompt Examples

## MMLU
You are a bot designed to answer questions by choosing between A, B, C, and D. Only one of these answers from the provided choices is correct. Reply with the letter of the correct answer.
Question: Alfred and Ben don't know each other but are each considering asking the lovely Charlene to the school prom. The probability that at least one of them will ask her is 0.72. The probability that they both ask her is 0.18. The probability that Alfred asks her is 0.6. What is the probability that Ben asks Charlene to the prom?
A: 0.78
B: 0.3
C: 0.24
D: 0.48
Answer: B

Question: A telephone survey of 400 registered voters showed that 256 had not yet made up their minds 1 month before the election. How sure can we be that between 60% and 68% of the electorate were still undecided at that time?
A: 2.4%
B: 8.0%
C: 64.0%
D: 90.4%
Answer:


You are a bot designed to answer questions by choosing between A, B, C, and D. Only one of these answers from the provided choices is correct. Reply with the letter of the correct answer.
Question: The procedure involving repeated presentation of a stimulus to the client until the attractiveness of that stimulus is reduced is best described as
A: stimulus satiation
B: response-prevention
C: flooding
D: implosion
Answer: A

Question: If, during a postexamination discussion with parents, a psychologist establishes that a child’s new pediatrician is apparently unaware of the child's history of brain damage. which is very important in understanding the problem situation, the psychologist should
A: tell the parents that he/she will inform the pediatrician
B: urge the parents to grant him/her permission to inform the pediatrician
C: cell the parents char be/she is legally obligated to inform the pediatrician
D: cell the parents that it is their responsibility to inform the pediatrician
Answer:

## Winogrande
You are a bot that is responsible for answering fill-in-the-blank questions. You are provided with a sentence and two possible fill-in-the-blank options. Your task is to return 1 or 2 as the answer.
Q: The store had 80 platters but only 2 bowls left in stock because the _ were in high demand.. 1 = platters and 2 = bowls. Answer:
2

Q: The smell in the kitchen of the home is unbearable, while the laundry room smells fine. The _ must have been cleaned longer ago.. 1 = kitchen and 2 = laundry room. Answer:



You are a bot that is responsible for answering fill-in-the-blank questions. You are provided with a sentence and two possible fill-in-the-blank options. Your task is to return 1 or 2 as the answer.
Q: John painted the pole red close to the color of the wall and painted the frame white and now the _ is similar.. 1 = frame and 2 = pole. Answer:
2

Q: Benjamin has a spouse and Kyle is single after being divorced, so _ is celebrating their independence this year.. 1 = Benjamin and 2 = Kyle. Answer:

## Arc Easy/Challenge
You are a bot designed to answer multiple choice questions. Given a series of options, you answer by choosing A, B, C, or D. Some examples are below.
Question: Which process uses carbon from the air to make food for plants?
A: growth
B: respiration
C: decomposition
D: photosynthesis
Answer: D

Question: An object composed mainly of ice is orbiting the Sun in an elliptical path. This object is most likely
A: a planet.
B: an asteroid.
C: a meteor.
D: a comet.
Answer:



You are a bot designed to answer multiple choice questions. Given a series of options, you answer by choosing A, B, C, or D. Some examples are below.
Question: Decomposers are important in the food chain because they
A: produce their own food using light from the Sun.
B: stop the flow of energy from one organism to another.
C: break down dead organisms and recycle nutrients into the soil.
D: are microscopic and other organisms cannot consume them.
Answer: C

Question: A wire is wrapped around a metal nail and connected to a battery. If the battery is active, the nail will
A: vibrate.
B: create sound.
C: produce heat.
D: become magnetic.
Answer:

## Humaneval
```
You are a code-writing bot. Given a function signature, and a docstring, complete the program body. Some examples are given below.
def sort_array(arr):
    """
    In this Kata, you have to sort an array of non-negative integers according to
    number of ones in their binary representation in ascending order.
    For similar number of ones, sort based on decimal value.

    It must be implemented like this:
    >>> sort_array([1, 5, 2, 3, 4]) == [1, 2, 3, 4, 5]
    >>> sort_array([-2, -3, -4, -5, -6]) == [-6, -5, -4, -3, -2]
    >>> sort_array([1, 0, 2, 3, 4]) [0, 1, 2, 3, 4]
    """
    return sorted(sorted(arr), key=lambda x: bin(x)[2:].count('1'))


def prime_fib(n: int):
    """
    prime_fib returns n-th number that is a Fibonacci number and it's also prime.
    >>> prime_fib(1)
    2
    >>> prime_fib(2)
    3
    >>> prime_fib(3)
    5
    >>> prime_fib(4)
    13
    >>> prime_fib(5)
    89
    """
```

```
# You are a code-writing bot. Given a function signature, and a docstring, complete the program body. Some examples are given below.
def unique(l: list):
    """Return sorted unique elements in a list
    >>> unique([5, 3, 5, 2, 3, 3, 9, 0, 123])
    [0, 2, 3, 5, 9, 123]
    """
    return sorted(list(set(l)))

def total_match(lst1, lst2):
    '''
    Write a function that accepts two lists of strings and returns the list that has 
    total number of chars in the all strings of the list less than the other list.

    if the two lists have the same number of chars, return the first list.

    Examples
    total_match([], []) ➞ []
    total_match(['hi', 'admin'], ['hI', 'Hi']) ➞ ['hI', 'Hi']
    total_match(['hi', 'admin'], ['hi', 'hi', 'admin', 'project']) ➞ ['hi', 'admin']
    total_match(['hi', 'admin'], ['hI', 'hi', 'hi']) ➞ ['hI', 'hi', 'hi']
    total_match(['4'], ['1', '2', '3', '4', '5']) ➞ ['4']
    '''
```

## TriviaQA

You are a trivia answering bot designed to answer questions. You are given a question and are supposed to output an answer in 1-3 words. Some examples are below.
Q: To what RAF base, near Wooton Bassett village, were the bodies of servicemen killed in Afghanistan formerly transported
A: LYNEHAM

Q: What star sign is Jamie Lee Curtis?
A:


You are a trivia answering bot designed to answer questions. You are given a question and are supposed to output an answer in 1-3 words. Some examples are below.
Q: Pre restraining order(s), who did People magazine name as their first "Sexiest Man Alive", in 1985?
A: Mel Gibson

Q: What id the name given to the study of birds?
A: