package org.dora.algorithm.geeksforgeek;

import org.dora.algorithm.datastructe.Node;

import java.util.*;

/**
 * @author dora
 * @date 2019/11/4
 */
public class MathematicalAlgorithm {

    public static void main(String[] args) {
        MathematicalAlgorithm algorithm = new MathematicalAlgorithm();
    }

    /**
     * 牛顿平方法来解决
     * 69. Sqrt(x)
     *
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        double precision = 0.00001;

        double result = x;

        while ((result * result - x) > precision) {
            result = (result + x / result) / 2;
        }
        return (int) result;
    }

    /**
     * todo 牵扯到格雷码 相关知识
     * 89. Gray Code
     * 格雷码
     *
     * @param n
     * @return
     */
    public List<Integer> grayCode(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        for (int i = 0; i < n; i++) {

        }
        return null;
    }

    /**
     * 91. Decode Ways
     *
     * @param s
     * @return
     */
    public int numDecodings(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        return 0;
    }

    /**
     * 93. Restore IP Addresses
     *
     * @param s
     * @return
     */
    public List<String> restoreIpAddresses(String s) {
        if (s == null || s.length() == 0) {
            return new ArrayList<>();
        }
        List<String> ans = new ArrayList<>();
        for (int a = 1; a <= 4; a++) {
            for (int b = 1; b <= 4; b++) {
                for (int c = 1; c <= 4; c++) {
                    for (int d = 1; d <= 4; d++) {
                        int len = a + b + c + d;
                        if (len == 12) {
                            int value1 = Integer.parseInt(s.substring(0, a));
                            int value2 = Integer.parseInt(s.substring(a, a + b));
                            int value3 = Integer.parseInt(s.substring(a + b, a + b + c));
                        }
                    }

                }

            }

        }
        return ans;
    }


    // ----------图理论graph----//

    /**
     * 128. Longest Consecutive Sequence
     *
     * @param nums
     * @return
     */
    public int longestConsecutive(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int result = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (map.containsKey(num)) {
                continue;
            }
            int leftEdge = map.getOrDefault(num - 1, 0);
            int rightEdge = map.getOrDefault(num + 1, 0);

            int val = leftEdge + rightEdge + 1;

            result = Math.max(result, val);

            map.put(num - leftEdge, val);
            map.put(num + rightEdge, val);
            map.put(num, val);
        }
        return result;
    }


    // -------位运算 Bits -----------//

    /**
     * 133. Clone Graph
     *
     * @param node
     * @return
     */
    public Node cloneGraph(Node node) {
        return null;
    }

    /**
     * 136. Single Number
     *
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        if (nums == null || nums.length == 0) {
            return Integer.MAX_VALUE;
        }
        int result = 0;

        for (int num : nums) {

            result ^= num;
        }
        return result;
    }

    /**
     * https://leetcode.com/problems/single-number-ii/discuss/43295/Detailed-explanation-and-generalization-of-the-bitwise-operation-method-for-single-numbers
     * 137. Single Number II
     *
     * @param nums
     * @return
     */
    public int singleNumberII(int[] nums) {
        return 0;
    }

    // ----------连续数字最大乘积--//

    /**
     * 149. Max Points on a Line
     *
     * @param points
     * @return
     */
    public int maxPoints(int[][] points) {
        return 0;
    }

    /**
     * 152. Maximum Product Subarray
     *
     * @param nums
     * @return
     */
    public int maxProduct(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int preMax = nums[0];
        int preMin = nums[0];
        int result = preMax;
        for (int i = 1; i < nums.length; i++) {
            int tmpMax = Math.max(Math.max(preMax * nums[i], preMin * nums[i]), nums[i]);
            int tmpMin = Math.min(Math.min(preMin * nums[i], preMax * nums[i]), nums[i]);
            result = Math.max(tmpMax, result);
            preMax = tmpMax;
            preMin = tmpMin;
        }
        return result;
    }

    // ----- //

    /**
     * 151. Reverse Words in a String
     *
     * @param s
     * @return
     */
    public String reverseWords(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        s = s.trim();
        String[] split = s.split(" ");
        StringBuilder builder = new StringBuilder();
        for (int i = split.length - 1; i >= 0; i--) {
            if (split[i].isEmpty()) {
                continue;
            }
            builder.append(split[i]);
            if (i > 0) {
                builder.append(" ");
            }
        }
        return builder.toString();
    }


    // ---- 摩尔投票法-- //

    /**
     * 166. Fraction to Recurring Decimal
     *
     * @param numerator
     * @param denominator
     * @return
     */
    public String fractionToDecimal(int numerator, int denominator) {
        return "";
    }


    // ---计算一个数 可以拥有0的数量--- //

    /**
     * 169. Majority Element
     *
     * @param nums
     * @return
     */
    public int majorityElement(int[] nums) {
        if (nums == null) {
            return -1;
        }
        int count = 1;
        int value = nums[0];
        for (int num : nums) {
            if (num == value) {
                count++;
            } else {
                count--;
            }
            if (count == 0) {
                value = num;
                count++;
            }
        }
        return value;
    }

    /**
     * 172. Factorial Trailing Zeroes
     * key case 判断拥有五的个数
     *
     * @param n
     * @return
     */
    public int trailingZeroes(int n) {
        if (n <= 0) {
            return 0;
        }
        int count = 0;
        while ((n / 5) != 0) {
            count += (n / 5);
            n /= 5;
        }
        return count;
    }

    /**
     * 168. Excel Sheet Column Title
     *
     * @param n
     * @return
     */
    public String convertToTitle(int n) {
        if (n <= 0) {
            return "";
        }
        StringBuilder builder = new StringBuilder();
        while (n != 0) {
            char val = (char) (((n - 1) % 26) + 'A');
            builder.append(val);
            n = (n - 1) / 26;
        }
        return builder.reverse().toString();
    }

    /**
     * 179. Largest Number
     *
     * @param nums
     * @return
     */
    public String largestNumber(int[] nums) {
        if (nums == null || nums.length == 0) {
            return "";
        }
        String[] strs = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            strs[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(strs, (o1, o2) -> {
            String tmp1 = o1 + o2;
            String tmp2 = o2 + o1;
            return tmp2.compareTo(tmp1);
        });
        if (strs[0].equals("0")) {
            return "0";
        }
        StringBuilder builder = new StringBuilder();
        for (String str : strs) {
            builder.append(str);
        }
        return builder.toString();
    }

    /**
     * 一个int数值 占据32位置
     * 必须把32位都挪移
     * 190. Reverse Bits
     *
     * @param n
     * @return
     */
    public int reverseBits(int n) {
        int result = 0;
        for (int i = 0; i < 32; i++) {

            result += n & 1;

            n >>= 1;

            if (i < 31) {
                result <<= 1;
            }
        }
        return result;
    }

    /**
     * 191. Number of 1 Bits
     *
     * @param n
     * @return
     */
    public int hammingWeight(int n) {
        int result = 0;
        while (n != 0) {
            result++;
            n = n & (n - 1);
        }
        return result;
    }

    /**
     * 202. Happy Number
     *
     * @param n
     * @return
     */
    public boolean isHappy(int n) {
        if (n <= 0) {
            return false;
        }
        Set<Integer> set = new HashSet<>();

        while (n != 0) {
            int tmp = n;

            int result = 0;

            while (tmp != 0) {
                int index = tmp % 10;

                result += index * index;

                tmp /= 10;
            }
            if (set.contains(result)) {
                return false;
            }
            if (result == 1) {
                return true;
            }

            set.add(result);

            n = result;
        }
        return false;
    }

    // ---城市天际线问题---- //

    /**
     * 204. Count Primes
     *
     * @param n
     * @return
     */
    public int countPrimes(int n) {
        if (n <= 0) {
            return 0;
        }
        int count = 0;
        for (int i = 0; i < Math.sqrt(n); i++) {

        }
        return 0;

    }


    // ---矩形面积问题-- //

    /**
     * 218. The Skyline Problem
     *
     * @param buildings
     * @return
     */
    public List<List<Integer>> getSkyline(int[][] buildings) {
        if (buildings == null || buildings.length == 0) {
            return new ArrayList<>();
        }
        return null;
    }

    /**
     * 223. Rectangle Area
     *
     * @param A
     * @param B
     * @param C
     * @param D
     * @param E
     * @param F
     * @param G
     * @param H
     * @return
     */
    public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        return 0;
    }

    /**
     * 229. Majority Element II
     * 摩尔投票法
     *
     * @param nums
     * @return
     */
    public List<Integer> majorityElementII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<Integer> ans = new ArrayList<>();
        int candidateA = nums[0];
        int candidateB = nums[0];

        int countA = 0;
        int countB = 0;
        for (int num : nums) {
            if (num == candidateA) {
                countA++;
                continue;
            }
            if (candidateB == num) {
                countB++;
                continue;
            }
            if (countA == 0) {
                candidateA = num;
                countA = 1;
                continue;
            }
            if (countB == 0) {
                candidateB = num;
                countB = 1;
                continue;
            }
            countA--;
            countB--;
        }
        countA = 0;
        countB = 0;
        for (int num : nums) {
            if (num == candidateA) {
                countA++;
            } else if (num == candidateB) {
                countB++;
            }
        }

        if (countA * 3 > nums.length) {
            ans.add(candidateA);
        }
        if (countB * 3 > nums.length) {
            ans.add(candidateB);
        }
        return ans;
    }

    /**
     * 231. Power of Two
     *
     * @param n
     * @return
     */
    public boolean isPowerOfTwo(int n) {
        if (n <= 0) {
            return false;
        }
        return (n & (n - 1)) == 0;
    }

    /**
     * todo 需要解决一个数字中 1的个数
     * 233. Number of Digit One
     *
     * @param n
     * @return
     */
    public int countDigitOne(int n) {
        return 0;
    }

    /**
     * 238. Product of Array Except Self
     *
     * @param nums
     * @return
     */
    public int[] productExceptSelf(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        int[] result = new int[nums.length];
        int base = 1;
        for (int i = 0; i < nums.length; i++) {
            result[i] = base;
            base *= nums[i];
        }
        base = 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            result[i] *= base;
            base *= nums[i];
        }
        return result;
    }

    /**
     * 241. Different Ways to Add Parentheses
     *
     * @param input
     * @return
     */
    public List<Integer> diffWaysToCompute(String input) {
        if (input == null || input.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> ops = new ArrayList<>();
        for (int i = 0; i < input.length(); i++) {
            int j = i;
            while (i < input.length() && Character.isDigit(input.charAt(i))) {
                i++;
            }
            ops.add(input.substring(j, i));
            if (i != input.length()) {
                ops.add(input.substring(i, i + 1));
            }
        }
        return intervalCompute(ops, 0, ops.size() - 1);
    }

    // ----递增子序列---//

    private List<Integer> intervalCompute(List<String> ops, int start, int end) {
        List<Integer> ans = new ArrayList<>();
        if (start == end) {
            ans.add(Integer.parseInt(ops.get(start)));
            return ans;
        }
        for (int i = start + 1; i <= end - 1; i = i + 2) {
            List<Integer> leftNums = intervalCompute(ops, start, i - 1);

            List<Integer> rightNums = intervalCompute(ops, i + 1, end);

            String sign = ops.get(i);

            for (Integer leftNum : leftNums) {
                for (Integer rightNum : rightNums) {
                    if (sign.equals("+")) {
                        ans.add(leftNum + rightNum);
                    } else if (sign.equals("-")) {
                        ans.add(leftNum - rightNum);
                    } else if (sign.equals("*")) {
                        ans.add(leftNum * rightNum);
                    } else if (sign.equals("/")) {
                        ans.add(leftNum / rightNum);
                    }
                }
            }
        }
        return ans;
    }

    /**
     * 300. Longest Increasing Subsequence
     *
     * @param nums
     * @return
     */
    public int lengthOfLIS(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int len = nums.length;
        int[] dp = new int[len];
        for (int i = 0; i < dp.length; i++) {
            dp[i] = 1;
        }
        for (int i = 1; i < len; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i] && dp[i] < dp[j] + 1) {
                    dp[i] = dp[j] + 1;
                }
            }
        }
        int result = 0;
        for (int i : dp) {
            result = Math.max(result, i);
        }
        return result;
    }

    /**
     * 260. Single Number III
     * key point:
     * 数字里面除了两个不同的数 其他数出现两次
     * 故将数组分成两个部分。
     *
     * @param nums
     * @return
     */
    public int[] singleNumberIII(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        int[] ans = new int[2];


        int value = 0;


        for (int num : nums) {
            value ^= num;
        }
//        int index = 0;
//        for (int i = 0; i < 32; i++) {
//
//            if ((value & (1 << i)) != 0) {
//                index = i;
//                break;
//            }
//        }
//        int base = 1 << index;

        // ？？？ 不懂为什么
        // Get its last set bit
        value &= -value;

        for (int num : nums) {

            int tmp = value & num;

            if (tmp != 0) {
                ans[0] ^= num;
            } else {
                ans[1] ^= num;
            }
        }
        return ans;

    }

    /**
     * 263. Ugly Number
     *
     * @param num
     * @return
     */
    public boolean isUgly(int num) {
        if (num <= 0) {
            return false;
        }
        if (num < 7) {
            return true;
        }
        for (int i = 2; i < 6 && num > 1; i++) {

            while (num % i == 0) {
                num /= i;
            }
        }
        return num == 1;
    }


    /**
     * todo
     * 264. Ugly Number II
     *
     * @param n
     * @return
     */
    public int nthUglyNumber(int n) {
        if (n <= 0) {
            return 0;
        }
        if (n < 7) {
            return n;
        }
        int[] result = new int[n];

        result[0] = 1;

        int index2 = 0;
        int index3 = 0;
        int index5 = 0;
        int index = 1;

        while (index < n) {
            int tmp = Math.min(result[index2] * 2, Math.min(result[index3] * 3, result[index5] * 5));

            if (tmp == result[index2] * 2) {
                index2++;
            }
            if (tmp == result[index3] * 3) {
                index3++;
            }
            if (tmp == result[index5] * 5) {
                index5++;
            }
            result[index++] = tmp;
        }
        return result[n - 1];
    }


    /**
     * 269 Alien Dictionary
     *
     * @param words: a list of words
     * @return: a string which is correct order
     */
    public String alienOrder(String[] words) {
        // Write your code here
        return "";
    }

    /**
     * https://leetcode.com/problems/h-index/
     * 274. H-Index
     *
     * @param citations
     * @return
     */
    public int hIndex(int[] citations) {
        if (citations == null || citations.length == 0) {
            return -1;
        }
        Arrays.sort(citations);
        List<Integer> result = new ArrayList<>();
        for (int i = citations.length - 1; i >= 0; i--) {
            result.add(citations[i]);
        }
        for (int i = 0; i < result.size(); i++) {
            if (i >= result.get(i)) {
                return i;
            }
        }
        return result.size();

    }


    /**
     * https://leetcode.com/problems/h-index-ii/
     *
     * @param citations
     * @return
     */
    public int hIndexII(int[] citations) {
        return -1;
    }


    /**
     * 277 Find the Celebrity
     *
     * @param n a party with n people
     * @return the celebrity's label or -1
     */
    public int findCelebrity(int n) {
        // Write your code here
        if (n < 0) {
            return -1;
        }
        int candidate = 0;
        for (int i = 0; i < n; i++) {
            if (knows(candidate, i)) {
                candidate = i;
            }
        }
        for (int i = candidate + 1; i < n; i++) {
            if (knows(candidate, i)) {
                return -1;
            }
        }
        for (int i = 0; i < candidate - 1; i++) {
            if (!knows(i, candidate) || knows(candidate, i)) {
                return -1;
            }
        }
        return candidate;
    }


    public boolean knows(int A, int B) {
        return true;
    }


}
