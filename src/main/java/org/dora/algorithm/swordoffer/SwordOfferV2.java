package org.dora.algorithm.swordoffer;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.ArrayList;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/1/29
 */
public class SwordOfferV2 {

    public static void main(String[] args) {
        SwordOfferV2 swordOfferV2 = new SwordOfferV2();
        int[] array = new int[]{1, 2, 3, 4, 5, 6, 7};
        swordOfferV2.reOrderArray(array);
    }


    /**
     * 二维数组中的查找
     *
     * @param target
     * @param array
     * @return
     */
    public boolean Find(int target, int[][] array) {
        if (array == null || array.length == 0) {
            return false;
        }
        int row = array.length;
        int column = array[0].length;
        int i = row - 1;
        int j = 0;
        while (i >= 0 && j < column) {
            int val = array[i][j];
            if (val == target) {
                return true;
            } else if (val < target) {
                j++;
            } else {
                i--;
            }
        }
        return false;
    }

    /**
     * 替换空格
     *
     * @param str
     * @return
     */
    public String replaceSpace(StringBuffer str) {
        if (str == null) {
            return "";
        }
//        String s = str.toString();
//        int blankCount = 0;
//        for (int i = 0; i < s.length(); i++) {
//            if (s.charAt(i) == ' ') {
//                blankCount++;
//            }
//        }
//        if (blankCount == 0) {
//            return s;
//        }
        StringBuilder builder = new StringBuilder();
        int len = str.length() - 1;
        while (len >= 0) {
            if (str.charAt(len) != ' ') {
                builder.append(str.charAt(len));
            } else {
                builder.append("02%");
            }
            len--;
        }
        return builder.reverse().toString();
    }

    public String replaceSpaceV2(StringBuffer str) {
        if (str == null) {
            return "";
        }

        int len = str.length();

        if (len == 0) {
            return "";
        }
        int blankWordCount = 0;
        for (int i = 0; i < len; i++) {
            if (str.charAt(i) == ' ') {
                blankWordCount++;
            }
        }
        if (blankWordCount == 0) {
            return str.toString();
        }
        int newWordCount = len - blankWordCount + blankWordCount * 3;

        char[] words = new char[newWordCount];

        int index = len - 1;

        int side = newWordCount - 1;

        while (index >= 0) {
            if (str.charAt(index) != ' ') {
                words[side--] = str.charAt(index);
            } else {
                words[side--] = '0';
                words[side--] = '2';
                words[side--] = '%';
            }
            index--;
        }
        return String.valueOf(words);
    }


    /**
     * 从尾到头打印链表
     *
     * @param listNode
     * @return
     */
    public ArrayList<Integer> printListFromTailToHeadV2(ListNode listNode) {
        if (listNode == null) {
            return new ArrayList<>();
        }
        ListNode prev = null;

        ListNode head = listNode;
        while (head != null) {
            ListNode tmp = head.next;
            head.next = prev;
            prev = head;
            head = tmp;
        }
        ArrayList<Integer> ans = new ArrayList<>();
        while (prev != null) {
            ans.add(prev.val);
            prev = prev.next;
        }
        return ans;
    }


    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {

        if (listNode == null) {
            return new ArrayList<>();
        }
        if (listNode.next == null) {
            ArrayList<Integer> ans = new ArrayList<>();
            ans.add(listNode.val);
            return ans;
        }
        ArrayList<Integer> integers = printListFromTailToHead(listNode.next);
        integers.add(listNode.val);
        return integers;

    }


    /**
     * 重建二叉树
     *
     * @param pre
     * @param in
     * @return
     */
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre == null || in == null) {
            return null;
        }
        if (pre.length != in.length) {
            return null;
        }
        return buildConstruct(0, pre, 0, in.length - 1, in);
    }

    private TreeNode buildConstruct(int preStart, int[] pre, int inStart, int inEnd, int[] in) {
        if (preStart < 0 || preStart >= pre.length || inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(pre[preStart]);
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (in[i] == root.val) {
                index = i;
                break;
            }
        }
        root.left = buildConstruct(preStart + 1, pre, inStart, index - 1, in);
        root.right = buildConstruct(preStart + index - inStart + 1, pre, index + 1, inEnd, in);
        return root;
    }


    /**
     * todo: 跑完全部测试用例
     * 旋转数组的最小数字
     *
     * @param array
     * @return
     */
    public int minNumberInRotateArray(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
            if (array[left] < array[right]) {
                return array[left];
            }
            int mid = left + (right - left) / 2;

            if (array[left] <= array[mid]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return array[left];
    }


    /**
     * [1,2,3,1,1,1,1,1,1]
     * <p>
     * [2,3,1,1,1]
     *
     * @param array
     * @return
     */
    public int minNumberInRotateArrayV2(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int left = 0;
        int right = array.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (array[mid] <= array[right]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return array[left];
    }


    /**
     * 斐波那切数列
     *
     * @param n
     * @return
     */
    public int Fibonacci(int n) {
        if (n == 0) {
            return 0;
        }
        if (n == 1) {
            return 1;
        }
        if (n == 2) {
            return 2;
        }
        return Fibonacci(n - 1) + Fibonacci(n - 2);
    }

    public int FibonacciV2(int n) {
        if (n == 0) {
            return 0;
        }
        if (n <= 2) {
            return 1;
        }
        int sum1 = 1;
        int sum2 = 1;
        int result = 0;
        for (int i = 3; i <= n; i++) {
            result = sum1 + sum2;
            sum1 = sum2;
            sum2 = result;
        }
        return result;
    }

    /**
     * 跳台阶
     *
     * @param target
     * @return
     */
    public int JumpFloor(int target) {
        if (target == 1) {
            return 1;
        }
        if (target == 2) {
            return 2;
        }
        return JumpFloor(target - 1) + JumpFloor(target - 2);
    }


    /**
     * 变态跳台阶
     *
     * @param target
     * @return
     */
    public int JumpFloorII(int target) {
        if (target <= 0) {
            return 0;
        }
        if (target == 1) {
            return 1;
        }
        if (target == 2) {
            return 2;
        }
        return 2 * JumpFloor(target - 1);
    }

    /**
     * 二进制中1的个数
     *
     * @param n
     * @return
     */
    public int NumberOf1(int n) {
        int result = 0;
        while (n != 0) {
            result++;
            n = n & (n - 1);
        }
        return result;
    }


    /**
     * 数值的整数次方
     *
     * @param base
     * @param exponent
     * @return
     */
    public double Power(double base, int exponent) {
        if (exponent == 0) {
            return 1;
        }
        if (exponent == 1) {
            return base;
        }
        if (exponent < 0) {
            exponent = -exponent;
            base = 1 / base;
        }
        return (exponent % 2 == 0) ? this.Power(base * base, exponent / 2) : base * this.Power(base * base, exponent / 2);
    }


    public double PowerV2(double base, int exponent) {
        if (exponent == 0) {
            return 1;
        }
        double result = 1.0;
        int absExponent = Math.abs(exponent);
        while (absExponent != 0) {
            if ((absExponent % 2) != 0) {
                result *= base;
            }
            base *= base;
            absExponent >>= 1;
        }
        return exponent < 0 ? 1 / result : result;
    }


    /**
     * 调整数组顺序使奇数位于偶数前面
     *
     * @param array
     */
    public void reOrderArray(int[] array) {
        if (array == null || array.length == 0) {
            return;
        }
        int len = array.length;
        int[] result = new int[len];
        int index = 0;
        for (int num : array) {
            if (num % 2 != 0) {
                result[index++] = num;
            }
        }
        for (int num : array) {
            if (num % 2 == 0) {
                result[index++] = num;
            }
        }
        array = result;
    }

    public void reOrderArrayV2(int[] array) {
        if (array == null || array.length == 0) {
            return;
        }
        for (int i = 0; i < array.length; i++) {
            for (int j = array.length - 1; j > i; j--) {
                if (array[j] % 2 == 1 && array[j - 1] % 2 == 0) {
                    swap(array, j, j - 1);
                }
            }
        }
    }

    private void swap(int[] nums, int i, int j) {
        int val = nums[i];
        nums[i] = nums[j];
        nums[j] = val;
    }


    /**
     * 1 - > 2 - > 3 - > 4
     * 2
     * 链表中倒数第k个结点
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode FindKthToTail(ListNode head, int k) {
        if (head == null || k <= 0) {
            return null;
        }
        ListNode fast = head;
        for (int i = 0; i < k - 1; i++) {
            fast = fast.next;
            if (fast == null) {
                return null;
            }
        }
        ListNode slow = head;
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }


    /**
     * 反转链表
     *
     * @param head
     * @return
     */
    public ListNode ReverseListV2(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode prev = null;
        ListNode current = head;
        while (current != null) {
            ListNode tmp = current.next;
            current.next = prev;
            prev = current;
            current = tmp;
        }
        return prev;
    }


    public ListNode ReverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode listNode = ReverseList(head.next);
        head.next.next = head;
        head.next = null;
        return listNode;
    }


    /**
     * 合并两个排序的链表
     *
     * @param list1
     * @param list2
     * @return
     */
    public ListNode Merge(ListNode list1, ListNode list2) {
        if (list1 == null && list2 == null) {
            return null;
        }
        if (list1 == null) {
            return list2;
        }
        if (list2 == null) {
            return list1;
        }
        if (list1.val <= list2.val) {
            list1.next = this.Merge(list1.next, list2);
            return list1;
        } else {
            list2.next = this.Merge(list1, list2.next);
            return list2;
        }
    }


    /**
     * 树的子结构
     *
     * @param root1
     * @param root2
     * @return
     */
    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null) {
            return false;
        }
        return isSubTree(root1, root2) || HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);

    }

    private boolean isSubTree(TreeNode root, TreeNode node) {
        if (root == null) {
            return true;
        }
        if (node == null) {
            return false;
        }
        if (root.val != node.val) {
            return false;
        }
        return isSubTree(root.left, node.left) && isSubTree(root.right, node.right);˚
    }


    /**
     * 树的反转
     *
     * @param root
     */
    public void Mirror(TreeNode root) {
        if (root == null) {
            return;
        }
        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;
        Mirror(root.left);
        Mirror(root.right);
    }


    /**
     * 打印螺旋矩阵
     *
     * @param matrix
     * @return
     */
    public ArrayList<Integer> printMatrix(int[][] matrix) {
        ArrayList<Integer> result = new ArrayList<>();
        if (matrix == null || matrix.length == 0) {
            return result;
        }
        int left = 0;
        int right = matrix[0].length - 1;
        int top = 0;
        int bottom = matrix.length - 1;
        while (left <= right && top <= bottom) {
            for (int j = left; j <= right; j++) {
                result.add(matrix[top][j]);
            }
            for (int i = top + 1; i <= bottom; i++) {
                result.add(matrix[i][right]);
            }
            if (top != bottom) {
                for (int j = right - 1; j >= left; j--) {
                    result.add(matrix[bottom][j]);
                }
            }
            if (left != right) {
                for (int i = bottom - 1; i > top; i--) {
                    result.add(matrix[i][left]);
                }
            }
            left++;
            top++;
            bottom--;
            right--;
        }
        return result;
    }


    /**
     * 栈的压入、弹出序列
     *
     * @param pushA
     * @param popA
     * @return
     */
    public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA == null || popA == null || pushA.length == 0 || popA.length == 0) {
            return false;
        }
        int index = 0;
        return false;

    }


}
