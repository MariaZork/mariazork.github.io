---
layout: post
title: Algorithms&#58; 
subtitle: Top 8 algorithmic approaches to solve problems with arrays
author: Maria Zorkaltseva
categories: [Data Structures and Algorithms]
tags: [leetcode, algorithms]
feature-img: "assets/img/sample_feature_img.png"
excerpt_begin_separator: <!--excerpt-->
excerpt_separator: <!--more-->
comments: true
---

<!--excerpt-->

I've tackled various problems with arrays from resources like LeetCode, HackerRank, GeeksForGeeks and in this post I made an attempt to structure approaches to solve problems with Arrays. The ability to quickly find an approach to solve an algorithmic problem will be very useful for passing interviews that contain coding questions. In this post, we will analyze some examples from Leetcode resource and estimate the algorithmic complexity of solutions from the point of Big O-notation.

![algorithms](/assets/img/2021-06-24-arrays-algorithmic-approaches/Competitive-Programming.jpg)
<!--more-->

For our examples I will use Python language. Usually to solve the problems with arrays in Python, we use **list** objects and it should be pointed out that actually its different data structure, but here for simplicity I will use the term **array**. Also I will consider only 1-dimensional arrays here to leave behind the scenes Depth First Search (DFS) and Breadth First Search (BFS) algorithms on Graph structures.

### 1. Sort and Then Do Something

Python standart functions for sorting use [TimSort](https://en.wikipedia.org/wiki/Timsort) which is hybrid of Merge and Insertion sorting algorithms. Average time performance of this algorithm is O(Nlog(N)) and worse space complexity is O(N). So, use this knowledge when deriving performance of your solutions.

#### [628. Maximum Product of Three Numbers](https://leetcode.com/problems/maximum-product-of-three-numbers/) (**Easy**)

```
Given an integer array nums, find three numbers whose product is maximum and return the maximum product.

Example 1:
Input: nums = [1,2,3]
Output: 6

Example 2:
Input: nums = [1,2,3,4]
Output: 24

Example 3:
Input: nums = [-1,-2,-3]
Output: -6
 
Constraints:
3 <= nums.length <= 104
-1000 <= nums[i] <= 1000
```

{% highlight python %}
def maximumProduct(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums.sort()
    n = len(nums)
    return max(nums[0]*nums[1]*nums[n-1], nums[n-3]*nums[n-2]*nums[n-1])
{% endhighlight %}

- Time complexity: O(N*log(N))
- Space complexity: O(log(N)). Sorting takes O(log(N)) space.

### 2. Divide and Conquer

#### [1470. Shuffle the Array](https://leetcode.com/problems/shuffle-the-array/) (**Easy**)

```
Given the array nums consisting of 2n elements in the form [x1,x2,...,xn,y1,y2,...,yn].
Return the array in the form [x1,y1,x2,y2,...,xn,yn].

Example 1:
Input: nums = [2,5,1,3,4,7], n = 3
Output: [2,3,5,4,1,7] 
Explanation: Since x1=2, x2=5, x3=1, y1=3, y2=4, y3=7 then the answer is [2,3,5,4,1,7].

Example 2:
Input: nums = [1,2,3,4,4,3,2,1], n = 4
Output: [1,4,2,3,3,2,4,1]

Example 3:
Input: nums = [1,1,2,2], n = 2
Output: [1,2,1,2]
 
Constraints:
1 <= n <= 500
nums.length == 2n
1 <= nums[i] <= 10^3
```

[Algorithm:](https://tutorialspoint.dev/algorithm/divide-and-conquer/shuffle-2n-integers-format-a1-b1-a2-b2-a3-b3-bn-without-using-extra-space)
Divide the given array into half (say arr1[] and arr2[]) and swap second half element of arr1[] with first half element of arr2[]. Recursively do this for arr1 and arr2.

1. Let the array be a1, a2, a3, a4, b1, b2, b3, b4
2. Split the array into two halves: a1, a2, a3, a4 : b1, b2, b3, b4
3. Exchange element around the center: exchange a3, a4 with b1, b2 correspondingly.
you get: a1, a2, b1, b2, a3, a4, b3, b4
4. Recursively spilt a1, a2, b1, b2 into a1, a2 : b1, b2
then split a3, a4, b3, b4 into a3, a4 : b3, b4.
5. Exchange elements around the center for each subarray we get:
a1, b1, a2, b2 and a3, b3, a4, b4.

**Note:** This solution only handles the case when n = 2i where i = 0, 1, 2, …etc.

{% highlight python %}
def shufleArray(a, f, l):
    # If only 2 element, return
    if (l - f == 1):
        return
  
    # Finding mid to divide the array 
    mid = int((f + l) / 2) 
  
    # Using temp for swapping first 
    # half of second array 
    temp = mid + 1
  
    # Mid is use for swapping second 
    # half for first array 
    mmid = int((f + mid) / 2) 
  
    # Swapping the element 
    for i in range(mmid + 1, mid + 1): 
        a[i], a[temp] = a[temp], a[i]
        temp += 1
  
    # Recursively doing for first 
    # half and second half 
    shufleArray(a, f, mid) 
    shufleArray(a, mid + 1, l)

a = [1, 3, 5, 7, 2, 4, 6, 8]  
n = len(a)  
shufleArray(a, 0, n - 1)
{% endhighlight %}

- Time complexity: O(N*log(N))
- Space complexity: O(N)

Here is O(N^2) time complexity solution which use brute force approach.

{% highlight python %}
def shuffleArray(a, n):
    # Rotate the element to the left
    i, q, k = 0, 1, n
    while(i < n):
        j = k  
        while(j > i + q):
            a[j - 1], a[j] = a[j], a[j - 1]
            j -= 1
        i += 1
        k += 1
        q += 1

a = [1, 3, 5, 7, 2, 4, 6, 8]  
n = len(a)
shuffleArray(a, int(n / 2))
{% endhighlight %}

- Time complexity: O(N^2)
- Space complexity: O(1)

#### Related problems from LeetCode

- [53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/) (Easy)
- [912. Sort an Array](https://leetcode.com/problems/sort-an-array/) (Medium)
- [932. Beautiful Array](https://leetcode.com/problems/beautiful-array/) (Medium)

### 3. Binary Search Algorithm

#### [1365. How Many Numbers are Smaller Than the Current Number](https://leetcode.com/problems/how-many-numbers-are-smaller-than-the-current-number/) (**Easy**)

```
Given the array nums, for each nums[i] find out how many numbers in the array are smaller than it. That is, for each nums[i] you have to count the number of valid j's such that j != i and nums[j] < nums[i].

Return the answer in an array.

Example 1:
Input: nums = [8,1,2,2,3]
Output: [4,0,1,1,3]
Explanation: 
For nums[0]=8 there exist four smaller numbers than it (1, 2, 2 and 3). 
For nums[1]=1 does not exist any smaller number than it.
For nums[2]=2 there exist one smaller number than it (1). 
For nums[3]=2 there exist one smaller number than it (1). 
For nums[4]=3 there exist three smaller numbers than it (1, 2 and 2).

Example 2:
Input: nums = [6,5,4,8]
Output: [2,1,0,3]

Example 3:
Input: nums = [7,7,7,7]
Output: [0,0,0,0]
 
Constraints:
2 <= nums.length <= 500
0 <= nums[i] <= 100
```

{% highlight python %}
def binarySearch(arr,target):
    low = 0
    high = len(arr)
    close = 0
    while low<=high:
        mid = low+(high-low)//2
        if arr[mid] == target:
            close = mid
            high = mid-1
        elif arr[mid]>target:
            high = mid-1
        else:
            low = mid+1
    return close

def smallerNumbersThanCurrent(nums: List[int]) -> List[int]:
    sortedNum = sorted(nums)
    res = []

    for num in nums:
        res.append(binarySearch(sortedNum,num))
    return res
{% endhighlight %}

- Time complexity: O(N*log(N))
- Space complexity: O(N)

#### [33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/) (**Medium**)

```
There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is rotated at an unknown pivot index k (0 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

Example 1:
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Example 2:
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1

Example 3:
Input: nums = [1], target = 0
Output: -1

Constraints:
1 <= nums.length <= 5000
-104 <= nums[i] <= 104
All values of nums are unique.
nums is guaranteed to be rotated at some pivot.
-104 <= target <= 104
```

{% highlight python %}
def search(nums: List[int], target: int) -> int:
    if not nums:
        return -1

    left, right = 0, len(nums)-1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    if target == nums[left]:
        return left
    return -1
{% endhighlight %}

- Time complexity: O(log(N))
- Space complexity: O(1)

### 4. Greedy

#### [1827. Minimum Operations to Make the Array Increasing](https://leetcode.com/problems/minimum-operations-to-make-the-array-increasing/) (Easy)

#### [1733. Minimum Number of People to Teach](https://leetcode.com/problems/minimum-number-of-people-to-teach/) (Medium)

#### Related problems

- [1558. Minimum Numbers of Function Calls to Make Target Array](https://leetcode.com/problems/minimum-numbers-of-function-calls-to-make-target-array) (Medium)
- [455. Assign Cookies](https://leetcode.com/problems/assign-cookies/) (Easy)
- [1798. Maximum Number of Consecutive Values You Can Make](https://leetcode.com/problems/maximum-number-of-consecutive-values-you-can-make) (Medium)

### 5. Hash Map Using

#### [1. Two Sum](https://leetcode.com/problems/two-sum/) (Easy)

```
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.

Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Output: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:
Input: nums = [3,3], target = 6
Output: [0,1]
 
Constraints:
2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
```

{% highlight python %}
def twoSum(nums: List[int], target: int) -> List[int]:
    d = dict()

    for i in range(len(nums)):
        complement = target - nums[i]

        if complement not in d.keys():
            d[nums[i]] = i
        else:
            return [d[complement], i]

print(twoSum([1,2,3], 5))
{% endhighlight %}

- Time complexity: O(N)
- Space complexity: O(N)

#### [525. Contiguous Array](https://leetcode.com/problems/contiguous-array/) (Medium)

```
Given a binary array nums, return the maximum length of a contiguous subarray with an equal number of 0 and 1.

Example 1:
Input: nums = [0,1]
Output: 2
Explanation: [0, 1] is the longest contiguous subarray with an equal number of 0 and 1.

Example 2:
Input: nums = [0,1,0]
Output: 2
Explanation: [0, 1] (or [1, 0]) is a longest contiguous subarray with equal number of 0 and 1. 

Constraints:
1 <= nums.length <= 105
nums[i] is either 0 or 1.
```

{% highlight python %}
def findMaxLength(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    hash_map = {}
    curr_sum = 0
    max_len = 0
    n = len(nums)

    for i in range(0, n):
        if (nums[i] == 0):
            nums[i] = -1
        else:
            nums[i] = 1

    for i in range(0, n):
        curr_sum = curr_sum + nums[i]
        if (curr_sum == 0):
            max_len = i + 1
        if curr_sum in hash_map:
            if max_len < i - hash_map[curr_sum]:
                max_len = i - hash_map[curr_sum]
        else:
            hash_map[curr_sum] = i

    return max_len
{% endhighlight %}

- Time complexity: O(N)
- Space complexity: O(N)

#### Related problems

- [594. Longest Harmonious Subsequence](https://leetcode.com/problems/longest-harmonious-subsequence) (Easy)
- [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/solution/) (Medium)
- [1711. Count Good Meals](https://leetcode.com/problems/count-good-meals/) (Medium)
- [822. Card Flipping Game](https://leetcode.com/problems/card-flipping-game/) (Medium)
- [554. Brick Wall](https://leetcode.com/problems/brick-wall/) (Medium)
- [454. 4SumII](https://leetcode.com/problems/4sum-ii) (Medium)
- [1726. Tuple with the same product](https://leetcode.com/problems/tuple-with-same-product) (Medium)
- [1743. Restore the Array from Adjacent Pairs](https://leetcode.com/problems/restore-the-array-from-adjacent-pairs) (Medium)
- [442. Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array) (Medium)

### 6. Dynamic Programming

#### [746. Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/) (Easy)

#### [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) (Easy)

#### [55. Jump Game](https://leetcode.com/problems/jump-game) (Medium)

### 7. Two Pointers, Sliding Window approach

#### [1089. Duplicate Zeros](https://leetcode.com/problems/duplicate-zeros) (Easy)

#### [11. Container With Most Water](https://leetcode.com/problems/beautiful-array/) (Medium)

#### [904. Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/) (Medium)

### 8. Stack

#### [682. Baseball Game](https://leetcode.com/problems/baseball-game/) (Easy)

```
You are keeping score for a baseball game with strange rules. 
The game consists of several rounds, where the scores of past rounds 
may affect future rounds' scores.

At the beginning of the game, you start with an empty record. 
You are given a list of strings ops, where ops[i] 
is the ith operation you must apply to the record and is one of the following:

An integer x - Record a new score of x.

"+" - Record a new score that is the sum of the previous two scores. 
It is guaranteed there will always be two previous scores.

"D" - Record a new score that is double the previous score. 
It is guaranteed there will always be a previous score.

"C" - Invalidate the previous score, removing it from the record. 
It is guaranteed there will always be a previous score.

Return the sum of all the scores on the record.

Example 1:
Input: ops = ["5","2","C","D","+"]
Output: 30
Explanation:
"5" - Add 5 to the record, record is now [5].
"2" - Add 2 to the record, record is now [5, 2].
"C" - Invalidate and remove the previous score, record is now [5].
"D" - Add 2 * 5 = 10 to the record, record is now [5, 10].
"+" - Add 5 + 10 = 15 to the record, record is now [5, 10, 15].
The total sum is 5 + 10 + 15 = 30.

Example 2:
Input: ops = ["5","-2","4","C","D","9","+","+"]
Output: 27
Explanation:
"5" - Add 5 to the record, record is now [5].
"-2" - Add -2 to the record, record is now [5, -2].
"4" - Add 4 to the record, record is now [5, -2, 4].
"C" - Invalidate and remove the previous score, record is now [5, -2].
"D" - Add 2 * -2 = -4 to the record, record is now [5, -2, -4].
"9" - Add 9 to the record, record is now [5, -2, -4, 9].
"+" - Add -4 + 9 = 5 to the record, record is now [5, -2, -4, 9, 5].
"+" - Add 9 + 5 = 14 to the record, record is now [5, -2, -4, 9, 5, 14].
The total sum is 5 + -2 + -4 + 9 + 5 + 14 = 27.

Example 3:
Input: ops = ["1"]
Output: 1
 

Constraints:
1 <= ops.length <= 1000
ops[i] is "C", "D", "+", or a string representing an integer 
in the range [-3 * 104, 3 * 104].
For operation "+", there will always be at least 
two previous scores on the record.
For operations "C" and "D", there will always be 
at least one previous score on the record.
```

{% highlight python %}
def calPoints(ops):
    stack = []
    for op in ops:
        if op == '+':
            stack.append(stack[-1] + stack[-2])
        elif op == 'C':
            stack.pop()
        elif op == 'D':
            stack.append(2 * stack[-1])
        else:
            stack.append(int(op))

    return sum(stack)
{% endhighlight %}

- Time complexity: O(N)
- Space complexity: O(N)

#### [739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures) (Medium)

#### Related problems

- [735. Asteroid Collision](https://leetcode.com/problems/asteroid-collision) (Medium)
