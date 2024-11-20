def zero_shot_prompt(question):
    zero_shot_prompt = f"""
    Solve the following question and provide the final answer in the format: 'Final Numeric Answer: (numeric value)'.
    Q: {question} \n
    A: 
    Final Numeric Answer: 
    \n\n
    """
    return zero_shot_prompt

def few_shot_prompt(question):
    few_shot_prompt = f"""
    Solve the following question and provide the final answer in the format: 'Final Numeric Answer: (numeric value)'. Here are some few shot examples to help generation: 
    -- BEGIN OF FEW SHOT EXAMPLES --
    1. 
    ```
    Sample Q: Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
    Sample A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11. 
    Sample Final Numeric Answer: 11
    \n\n
    ```
    2. 
    ```
    Sample Q: Julie, Micah, and Mitchell sold 32 glasses of lemonade at their lemonade stand. Julie sold 14 glasses and the boys sold an equal number of glasses. How many more glasses did Julie sell than Micah?
    Sample A: Micah and Mitchell sold 32 - 14 = <<32-14=18>>18 glasses. They each sold 18/2 = <<18/2=9>>9 glasses. Julie sold 14 - 9 = <<14-9=5>>5 glasses more than Micah
    Sample Final Numeric Answer: 5
    \n\n
    ```
    3. 
    ```
    Sample Q: Two sports coaches went shopping together. The baseball coach bought 9 new baseballs for $3 each. The basketball coach bought 8 new basketballs for $14 each. How much more did the basketball coach spend than the baseball coach?
    Sample A: The cost of the baseballs is 9 × $3 = $<<9*3=27>>27. The cost of the basketballs is 8 × $14 = $<<8*14=112>>112. Basketballs cost $112 − $27 = $85 more. 
    Sample Final Numeric Answer: 85
    \n\n
    ```
    -- END OF FEW SHOT EXAMPLES --
    Q: {question} \n
    A: 
    Final Numeric Answer: 
    \n\n
    """
    return few_shot_prompt

def cot_prompt(question):
    cot_prompt = f"""
    Solve the following question and provide the final answer in the format: 'Final Numeric Answer: (numeric value)'. Let's think step by step, ensure you use logical reasoning steps to arrive at your solution. Here are some few shot examples to help generation: 
    -- BEGIN OF FEW SHOT EXAMPLES --
    1. 
    ```
    Sample Q: Charisma works for 8 hours every day.  She has a timer to remind her to get up and walk for 5 minutes every hour she’s at work.  After 5 days at the office, how many minutes has she walked?
    Sample A: Let's break it down step by step. Charisma walks for 5 minutes every hour, and she works for 8 hours a day. So, she walks for: 5 minutes/hour × 8 hours/day = 40 minutes/day. She works for 5 days, so she walks for: 40 minutes/day × 5 days = 200 minutes
    Sample Final Numeric Answer: 200
    \n\n
    ```
    2. 
    ```
    Sample Q: Julie, Micah, and Mitchell sold 32 glasses of lemonade at their lemonade stand. Julie sold 14 glasses and the boys sold an equal number of glasses. How many more glasses did Julie sell than Micah?
    Sample A: Let's break it down step by step! We know that Julie sold 14 glasses of lemonade. The total number of glasses sold is 32. The boys, Micah and Mitchell, sold an equal number of glasses. So, they sold a total of: 32 - 14 = 18 glasses. Since they sold an equal number of glasses, we divide 18 by 2 to find out how many glasses each boy sold: 18 ÷ 2 = 9. Now, we can find out how many more glasses Julie sold than Micah: Julie sold 14 glasses, and Micah sold 9 glasses. To find the difference, we subtract: 14 - 9 = 5. So, Julie sold 5 more glasses than Micah.
    Sample Final Numeric Answer: 5
    \n\n
    ```
    3. 
    ```
    Sample Q: Two sports coaches went shopping together. The baseball coach bought 9 new baseballs for $3 each. The basketball coach bought 8 new basketballs for $14 each. How much more did the basketball coach spend than the baseball coach?
    Sample A: Let's break it down step by step. The baseball coach bought 9 new baseballs for $3 each, so the total cost is: 9 x $3 = $27. The basketball coach bought 8 new basketballs for $14 each, so the total cost is: 8 x $14 = $112. Now, let's find out how much more the basketball coach spent: $112 (basketball coach) - $27 (baseball coach) = $85.
    Sample Final Numeric Answer: 85
    \n\n
    ```
    -- END OF FEW SHOT EXAMPLES --
    Q: {question} \n
    A: 
    Final Numeric Answer: 
    \n\n
    """
    return cot_prompt

def sc_prompt(question):
    sc_prompt = f"""
    Solve the following question and provide the final answer in the format: 'Final Numeric Answer: (numeric value)'. Let's think step by step, ensure you use logical reasoning steps to arrive at your solution. Here are some few shot examples to help generation: 
    -- BEGIN OF FEW SHOT EXAMPLES --
    1. 
    ```
    Sample Q: Charisma works for 8 hours every day.  She has a timer to remind her to get up and walk for 5 minutes every hour she’s at work.  After 5 days at the office, how many minutes has she walked?
    Sample A: Let's break it down step by step. Charisma walks for 5 minutes every hour, and she works for 8 hours a day. So, she walks for: 5 minutes/hour × 8 hours/day = 40 minutes/day. She works for 5 days, so she walks for: 40 minutes/day × 5 days = 200 minutes
    Sample Final Numeric Answer: 200
    \n\n
    ```
    2. 
    ```
    Sample Q: Julie, Micah, and Mitchell sold 32 glasses of lemonade at their lemonade stand. Julie sold 14 glasses and the boys sold an equal number of glasses. How many more glasses did Julie sell than Micah?
    Sample A: Let's break it down step by step! We know that Julie sold 14 glasses of lemonade. The total number of glasses sold is 32. The boys, Micah and Mitchell, sold an equal number of glasses. So, they sold a total of: 32 - 14 = 18 glasses. Since they sold an equal number of glasses, we divide 18 by 2 to find out how many glasses each boy sold: 18 ÷ 2 = 9. Now, we can find out how many more glasses Julie sold than Micah: Julie sold 14 glasses, and Micah sold 9 glasses. To find the difference, we subtract: 14 - 9 = 5. So, Julie sold 5 more glasses than Micah.
    Sample Final Numeric Answer: 5
    \n\n
    ```
    3. 
    ```
    Sample Q: Two sports coaches went shopping together. The baseball coach bought 9 new baseballs for $3 each. The basketball coach bought 8 new basketballs for $14 each. How much more did the basketball coach spend than the baseball coach?
    Sample A: Let's break it down step by step. The baseball coach bought 9 new baseballs for $3 each, so the total cost is: 9 x $3 = $27. The basketball coach bought 8 new basketballs for $14 each, so the total cost is: 8 x $14 = $112. Now, let's find out how much more the basketball coach spent: $112 (basketball coach) - $27 (baseball coach) = $85.
    Sample Final Numeric Answer: 85
    \n\n
    ```
    -- END OF FEW SHOT EXAMPLES --
    Q: {question} \n
    A: 
    Final Numeric Answer: 
    \n\n
    """
    return sc_prompt

def get_prompt(strategy, question):
    if strategy == 'zero-shot':
        return zero_shot_prompt(question)
    elif strategy == 'few-shot':
        return few_shot_prompt(question)
    elif strategy == 'cot':
        return cot_prompt(question)
    elif strategy == 'sc':
        return sc_prompt(question)
    else:
        raise ValueError('Invalid')