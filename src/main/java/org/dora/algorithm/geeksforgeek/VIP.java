package org.dora.algorithm.geeksforgeek;

import org.dora.algorithm.datastructe.Interval;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author dora
 * @date 2019/11/29
 */
public class VIP {


    /**
     * 250. Count Univalue Subtrees
     *
     * @param root
     * @return
     */
    int subCount = 0;

    public static void main(String[] args) {
        VIP vip = new VIP();
        TreeNode root = new TreeNode(3);

        TreeNode right = new TreeNode(4);
        root.right = right;

        TreeNode left = new TreeNode(1);

        left.right = new TreeNode(2);

        root.left = left;

        double target = 1.275000;
        vip.closestKValuesV2(root, target, 2);
    }

    /**
     * 151 Binary Tree Upside Down Medium
     */
    public TreeNode upsizeDownBinaryTree(TreeNode root) {
        if (root == null || root.left == null) {
            return root;
        }
        TreeNode left = root.left;
        TreeNode right = root.right;
        TreeNode ans = this.upsizeDownBinaryTree(left);
        left.left = right;
        left.right = root;
        root.left = null;
        root.right = null;
        return ans;

    }

    /**
     * 161 One Edit Distance
     *
     * @param s: a string
     * @param t: a string
     * @return: true if they are both one edit distance apart or false
     */
    public boolean isOneEditDistance(String s, String t) {
        // write your code here
        if (s == null || t == null) {
            return false;
        }
        int m = s.length();

        int n = t.length();

        if (m > n) {
            return isOneEditDistance(t, s);
        }
        int diff = Math.abs(m - n);
        if (diff > 1) {
            return false;
        }
        for (int i = 0; i < m; i++) {
            if (s.charAt(i) != t.charAt(i)) {
                if (diff == 0) {
                    return s.charAt(i + 1) == t.charAt(i + 1);
                } else {
                    return s.charAt(i) == t.charAt(i + 1);
                }
            }
        }
        return m != n;
    }

    /**
     * todo
     * 163 Missing Ranges
     *
     * @param nums:  a sorted integer array
     * @param lower: An integer
     * @param upper: An integer
     * @return: a list of its missing ranges
     */
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        List<String> ans = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            ans.add(getRange(lower, upper));
            return ans;
        }
        if (nums[0] > lower) {
            ans.add(getRange(lower, nums[0] - 1));
        }
        for (int i = 1; i < nums.length; i++) {
            int diff = nums[i] - nums[i - 1];
            if (diff > 1) {
                ans.add(getRange(nums[i - 1] + 1, nums[i] - 1));
            }
        }
        if (nums[nums.length - 1] < upper) {
            ans.add(getRange(nums[nums.length - 1] + 1, upper));
        }
        return ans;
    }

    private String getRange(int start, int end) {
        if (start == end) {
            return String.valueOf(start);
        }
        return start + "->" + end;
    }

    public List<String> findMissingRangesV2(int[] nums, int lower, int upper) {
        List<String> ans = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            ans.add(getRange(lower, upper));
            return ans;
        }
        if (nums[0] > lower) {
            ans.add(getRange(lower, nums[0] - 1));
        }
        int prev = nums[0];
        for (int i = 1; i <= nums.length; i++) {
            int cur = i == nums.length ? upper + 1 : nums[i];
            long diff = (long) cur - (long) prev;
            if (diff > 1) {
                ans.add(getRange(prev + 1, cur - 1));
            }
            prev = cur;
        }
        return ans;
    }

    /**
     * 解题思路 单纯构造一个hash表
     * 247 Strobogrammatic Number II
     *
     * @param num
     * @return
     */
    public boolean isStrobogrammatic(String num) {
        // write your code here
        if (num == null || num.length() == 0) {
            return false;
        }
        HashMap<Character, Character> map = new HashMap<>();
        map.put('0', '0');
        map.put('1', '1');
        map.put('6', '9');
        map.put('8', '8');
        map.put('9', '6');
        for (int i = 0; i <= num.length() / 2; i++) {
            Character character = map.getOrDefault(num.charAt(i), ' ');
            if (character.equals(num.charAt(num.length() - 1 - i))) {
                return false;
            }
        }
        return true;
    }

    /**
     * 247 Strobogrammatic Number II
     *
     * @param n: the length of strobogrammatic number
     * @return: All strobogrammatic numbers
     */
    public List<String> findStrobogrammatic(int n) {
        // write your code here
        if (n <= 0) {
            return new ArrayList<>();
        }
        return intervalStrobogrammatic(n, n);

    }

    private List<String> intervalStrobogrammatic(int n, int m) {
        if (n == 0) {
            return Arrays.asList("");
        }
        if (n == 1) {
            return Arrays.asList("0", "1", "8");
        }
        List<String> list = this.intervalStrobogrammatic(n - 2, m);
        List<String> result = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) {
            String item = list.get(i);

            if (n != m) {
                result.add("0" + item + "0");
            }

            result.add("1" + item + "1");
            result.add("6" + item + "9");
            result.add("8" + item + "8");
            result.add("9" + item + "6");
        }
        return result;
    }

    /**
     * 247 Strobogrammatic Number II
     *
     * @param n
     * @return
     */
    public List<String> findStrobogrammaticV2(int n) {
        List<String> result = new ArrayList<>();
        if (n == 0) {
            return new ArrayList<>();
        }
        LinkedList<String> queue = new LinkedList<>();

        //base case for even number
        String[] candidates = new String[]{"11", "69", "96", "88", "00"};

        String[] oddBaseCase = new String[]{"0", "1", "8"};

        List<String> list = n % 2 == 0 ? Arrays.asList(candidates) : Arrays.asList(oddBaseCase);
        queue.addAll(list);
        while (!queue.isEmpty()) {
            String s = queue.poll();
            if (s.length() == n && (s.length() == 1 || s.charAt(0) != '0')) {
                result.add(s);
            }
            if (s.length() > n) {
                return result;
            }
            for (int i = 0; i < candidates.length; i++) {
                queue.offer(candidates[i].charAt(0) + s + candidates[i].charAt(1));
            }
        }
        return result;
    }

    /**
     * 248 Strobogrammatic Number III
     *
     * @param low
     * @param high
     * @return
     */
    public int strobogrammaticInRange(String low, String high) {
        if (low == null || high == null) {
            return 0;
        }
        int[] result = new int[]{0};
        for (int i = low.length(); i <= high.length(); i++) {
            compute(i, "", low, high, result);
            compute(i, "0", low, high, result);
            compute(i, "1", low, high, result);
            compute(i, "8", low, high, result);
        }
        return result[0];
    }

    private void compute(int len, String item, String low, String high, int[] result) {
        if (item.length() >= len) {
            if (item.length() > len && (len != -1 && item.charAt(0) == '0')) {
                return;
            }
            if ((item.length() == low.length() && item.compareTo(low) < 0) || (item.length() == high.length() && item.compareTo(high) > 0)) {
                return;
            }
            int count = result[0];
            result[0] = ++count;
        }
        compute(len, "0" + item + "0", low, high, result);
        compute(len, "1" + item + "1", low, high, result);
        compute(len, "6" + item + "9", low, high, result);
        compute(len, "8" + item + "8", low, high, result);
        compute(len, "9" + item + "6", low, high, result);
    }

    /**
     * 248 Strobogrammatic Number III
     *
     * @param low
     * @param high
     * @return
     */
    public int strobogrammaticInRangeV2(String low, String high) {
        int[] result = new int[]{0};
        intervalCountInRange("", low, high, result);
        intervalCountInRange("0", low, high, result);
        intervalCountInRange("1", low, high, result);
        intervalCountInRange("8", low, high, result);
        return result[0];
    }

    private void intervalCountInRange(String item, String low, String high, int[] result) {
        int len = item.length();
        boolean correctRange = len <= low.length() && len <= high.length();

        if (correctRange) {
            int count = result[0];
            if (len == high.length() && item.compareTo(high) > 0) {
                return;
            }
            if (!(len > 1 && item.charAt(0) == '0') && !(len == low.length() && item.compareTo(low) > 0)) {
                ++count;
                result[0] = count;
            }

        }
        if (len + 2 > high.length()) {
            return;
        }
        this.intervalCountInRange("0" + item + "0", low, high, result);
        this.intervalCountInRange("1" + item + "1", low, high, result);
        this.intervalCountInRange("6" + item + "9", low, high, result);
        this.intervalCountInRange("8" + item + "8", low, high, result);
        this.intervalCountInRange("9" + item + "6", low, high, result);
    }

    /**
     * 249. Group Shifted Strings
     *
     * @param strings
     * @return
     */
    List<List<String>> groupStrings(List<String> strings) {
        return null;
    }

    /**
     * 250. Count Univalue Subtrees
     *
     * @param root
     * @return
     */
    public int countUnivalSubtrees(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (this.intervalSubTree(root, root.val)) {
            subCount++;
        }
        countUnivalSubtrees(root.left);
        countUnivalSubtrees(root.right);
        return subCount;
    }

    private boolean intervalSubTree(TreeNode root, int val) {
        if (root == null) {
            return true;
        }
        if (root.val == val) {
            return intervalSubTree(root.left, val) && intervalSubTree(root.right, val);
        }
        return false;
    }

    /**
     * @param root
     * @return
     */
    public int countUnivalSubtreesV2(TreeNode root) {
        if (root == null) {
            return 0;
        }
        intervalSubTreeV2(root, -1);
        return subCount;
    }

    private boolean intervalSubTreeV2(TreeNode root, int val) {
        if (root == null) {
            return true;
        }
        if (!(intervalSubTree(root.left, root.val) && intervalSubTree(root.right, root.val))) {
            return false;
        }
        subCount++;
        return root.val == val;
    }

    /**
     * 252 Meeting Rooms
     *
     * @param intervals: an array of meeting time intervals
     * @return: if a person could attend all meetings
     */
    public boolean canAttendMeetings(List<Interval> intervals) {
        if (intervals == null || intervals.isEmpty()) {
            return true;
        }
        Interval[] array = intervals.toArray(new Interval[]{});
        Arrays.sort(array, Comparator.comparingInt((Interval o) -> {
            return o.start;
        }));
        // Write your code here
        for (int i = 1; i < array.length; i++) {
            Interval pre = array[i - 1];
            Interval now = array[i];
            if (now.start < pre.end) {
                return false;
            }
        }
        return true;
    }

    /**
     * 253 Meeting Rooms II
     *
     * @param intervals: an array of meeting time intervals
     * @return: the minimum number of conference rooms required
     */
    public int minMeetingRooms(List<Interval> intervals) {
        // Write your code here
        if (intervals == null || intervals.isEmpty()) {
            return -1;
        }
        int len = intervals.size();
        int[] begin = new int[len];
        int[] end = new int[len];
        for (int i = 0; i < intervals.size(); i++) {
            Interval interval = intervals.get(i);
            begin[i] = interval.start;
            end[i] = interval.end;
        }
        Arrays.sort(begin);
        Arrays.sort(end);
        int result = 0;
        int pre = 0;
        for (int i = 1; i < len; i++) {
            result++;
            if (begin[i] >= end[pre]) {
                result--;
                pre++;
            }
        }
        return result;
    }

    /**
     * 254. Factor Combinations
     *
     * @param n
     * @return
     */
    List<List<Integer>> getFactors(int n) {
        if (n <= 1) {
            return new ArrayList<>();
        }
        if (n % 2 == 1) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();

        intervalFactors(ans, new ArrayList<>(), 2, n);

        return ans;
    }

    private void intervalFactors(List<List<Integer>> ans, List<Integer> integers, int start, int end) {
        if (end == 1) {
            if (integers.size() > 1) {
                ans.add(new ArrayList<>(integers));
            }
            return;
        }
        for (int i = start; i <= end; i++) {
            if (end % i == 0) {
                integers.add(i);
                intervalFactors(ans, integers, i, end / i);
                integers.remove(integers.size() - 1);
            }
        }
    }

    /**
     * 255. Verify Preorder Sequence in Binary Search
     *
     * @param preorder
     * @return
     */
    public boolean verifyPreorder(List<Integer> preorder) {
        if (preorder == null || preorder.isEmpty()) {
            return true;
        }
        int pre = Integer.MIN_VALUE;
        Stack<Integer> stack = new Stack<>();
        for (Integer item : preorder) {
            if (item < pre) {
                return false;
            }
            while (!stack.isEmpty() && item > stack.peek()) {
                pre = stack.pop();
            }
            stack.push(item);
        }
        return true;
    }


    public boolean verifyPreorderV2(int[] preorder) {
        if (preorder == null || preorder.length == 0) {
            return true;
        }
        int index = -1;
        int pre = Integer.MIN_VALUE;
        for (int i = 0; i < preorder.length; i++) {
            Integer item = preorder[i];
            if (item < pre) {
                return false;
            }
            while (index >= 0 && preorder[index] < item) {
                pre = preorder[index];
                index--;
            }
            preorder[++index] = item;
        }
        return true;
    }


    /**
     * 256 Paint House
     *
     * @param costs
     * @return
     */
    public int minCost(int[][] costs) {
        // write your code here
        if (costs == null || costs.length == 0) {
            return 0;
        }
        int row = costs.length;
        for (int i = 1; i < costs.length; i++) {
            costs[i][0] += Math.min(costs[i - 1][1], costs[i - 1][2]);
            costs[i][1] += Math.min(costs[i - 1][0], costs[i - 1][2]);
            costs[i][2] += Math.min(costs[i - 1][0], costs[i - 1][1]);
        }
        return Math.min(Math.min(costs[row - 1][0], costs[row - 1][1]), costs[row - 1][2]);
    }


    /**
     * 259 3Sum Smaller
     * key point:
     * 1、要求 i + j + k < target
     * 2、求满足条件1的个数
     * 3、为了找出个数 满足条件1的时候 直接 k - j
     * https://www.lintcode.com/problem/3sum-smaller/description
     *
     * @param nums:   an array of n integers
     * @param target: a target
     * @return: the number of index triplets satisfy the condition nums[i] + nums[j] + nums[k] < target
     */
    public int threeSumSmaller(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int result = 0;
        int len = nums.length;
        Arrays.sort(nums);
        for (int i = 0; i < len - 2; i++) {
            int left = i + 1;
            int right = len - 1;
            while (left < right) {
                int value = nums[i] + nums[left] + nums[right];
                if (value < target) {
                    result += right - left;
                    left++;
                } else {
                    right--;
                }
            }
        }
        return result;
    }


    /**
     * 无向图
     * https://github.com/grandyang/leetcode/issues/261
     * 261. Graph Valid Tree
     *
     * @param n
     * @param edges
     * @return
     */
    public boolean validTree(int n, int[][] edges) {
        return false;
    }


    /**
     * 266 Palindrome Permutation
     *
     * @param s: the given string
     * @return: if a permutation of the string could form a palindrome
     */
    public boolean canPermutePalindrome(String s) {
        // write your code here
        if (s == null || s.isEmpty()) {
            return true;
        }

        HashMap<Character, Integer> map = new HashMap<>();
        Set<Character> sets = new HashSet<>();
        for (char c : s.toCharArray()) {
            Integer count = map.getOrDefault(c, 0);

            count++;

            map.put(c, count);

            sets.add(c);
        }

        boolean existOdd = false;

        for (Character item : sets) {
            Integer integer = map.get(item);

            boolean oddNum = integer % 2 != 0;

            if (existOdd && oddNum) {
                return false;
            }
            if (oddNum) {
                existOdd = true;
            }
        }
        return true;
    }

    /**
     * 267 * Palindrome Permutation II
     *
     * @param s
     * @return
     */
    public List<String> generatePalindromes(String s) {
        // write your code here
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        HashMap<Character, Integer> map = new HashMap<>();
        char[] chars = s.toCharArray();

        int oddCount = 0;

        for (char tmp : chars) {

            Integer count = map.getOrDefault(tmp, 0);

            count++;

            map.put(tmp, count);

            oddCount += (count % 2 != 0 ? 1 : -1);

        }
        List<String> ans = new ArrayList<>();

        if (oddCount > 1) {
            return ans;
        }
        List<Character> tmp = new ArrayList<>();

        String mid = "";
        for (Map.Entry<Character, Integer> entry : map.entrySet()) {

            Character key = entry.getKey();

            Integer count = entry.getValue();


            if (count % 2 != 0) {
                mid = String.valueOf(key);
            }
            for (int j = 0; j < count / 2; j++) {
                tmp.add(key);
            }
        }
        boolean[] used = new boolean[tmp.size()];

        intervalPalindrome(mid, used, tmp, new StringBuilder(), ans);

        return ans;
    }

    private void intervalPalindrome(String mid, boolean[] used, List<Character> tmp, StringBuilder builder, List<String> ans) {
        if (builder.length() == tmp.size()) {
            String item = builder.toString() + mid + builder.reverse().toString();
            ans.add(item);
            builder.reverse();
            return;
        }
        for (int i = 0; i < tmp.size(); i++) {
            if (i > 0 && (tmp.get(i).equals(tmp.get(i - 1))) && !used[i - 1]) {
                continue;
            }
            if (!used[i]) {
                used[i] = true;
                builder.append(tmp.get(i));
                intervalPalindrome(mid, used, tmp, builder, ans);
                used[i] = false;
                builder.deleteCharAt(builder.length() - 1);
            }
        }
    }


    /**
     * 271 Encode and Decode Strings
     *
     * @param strs: a list of strings
     * @return: encodes a list of strings to a single string.
     */
    public String encode(List<String> strs) {
        // write your code here
        if (strs == null || strs.isEmpty()) {
            return "";
        }
        StringBuilder builder = new StringBuilder();

        for (int i = 0; i < strs.size(); i++) {
            String item = strs.get(i);

            if (":".equals(item)) {
                item += "::";
            } else {
                item += ":";
            }
            if (i < strs.size() - 1) {
                item += ";";
            }
            builder.append(item);
        }
        return builder.toString();
    }

    /**
     * @param str: A string
     * @return: dcodes a single string to a list of strings
     */
    public List<String> decode(String str) {
        // write your code here
        if (str == null || str.isEmpty()) {
            return new ArrayList<>();
        }
        String[] split = str.split(";");
        List<String> ans = new ArrayList<>();
        for (String item : split) {
            if (item.startsWith(":")) {
                ans.add(":");
            } else {
                item = item.substring(0, item.length() - 1);
                ans.add(item);
            }
        }
        return ans;
    }

    /**
     * 272 Closest Binary Search Tree Value II
     *
     * @param root:   the given BST
     * @param target: the given target
     * @param k:      the given k
     * @return: k values in the BST that are closest to the target
     */
    public List<Integer> closestKValues(TreeNode root, double target, int k) {
        // write your code here
        if (root == null) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>();


        List<Integer> values = new ArrayList<>();

        inorderTraversal(root, values);

        int index = 0;

        int n = values.size();

        while (index < n) {
            Integer integer = values.get(index);
            if (integer >= target) {
                break;
            }
            index++;
        }
        if (index >= n) {
            return values.subList(n - k, n);
        }

        int left = index - 1;
        int right = index;

        for (int i = 0; i < k; i++) {
            if (left >= 0 && (right >= n || target - values.get(left) < values.get(right) - target)) {
                result.add(values.get(left));
                left--;
            } else {
                result.add(values.get(right));
                right++;
            }
        }
        return result;


    }

    private void inorderTraversal(TreeNode root, List<Integer> valus) {
        if (root == null) {
            return;
        }
        inorderTraversal(root.left, valus);
        valus.add(root.val);
        inorderTraversal(root.right, valus);
    }


    public List<Integer> closestKValuesV2(TreeNode root, double target, int k) {
        if (root == null) {
            return new ArrayList<>();
        }
        TreeNode node = root;
        Stack<TreeNode> prev = new Stack<>();
        Stack<TreeNode> next = new Stack<>();
        while (node != null) {
            if (node.val < target) {
                prev.push(node);
                node = node.right;
            } else {
                next.push(node);
                node = node.left;
            }
        }
        List<Integer> result = new LinkedList<>();
        while (result.size() < k) {
            double distPrev = prev.isEmpty() ? Double.MAX_VALUE : Math.abs(target - prev.peek().val);
            double distNext = next.isEmpty() ? Double.MAX_VALUE : Math.abs(target - next.peek().val);

            if (distPrev < distNext) {
                result.add(0, prev.peek().val);

                getPrev(prev);
            } else {
                result.add(next.peek().val);
                getNext(next);
            }
        }
        return result;

    }

    private void getNext(Stack<TreeNode> next) {
        TreeNode r = next.pop().right;

        while (r != null) {
            next.push(r);
            r = r.left;
        }
    }

    private void getPrev(Stack<TreeNode> prev) {
        TreeNode l = prev.pop().left;
        while (l != null) {
            prev.push(l);
            l = l.right;
        }
    }


}
