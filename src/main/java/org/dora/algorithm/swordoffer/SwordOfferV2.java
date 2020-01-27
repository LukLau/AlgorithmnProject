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
        int[] array = new int[]{3, 4, 5, 1, 2};
        swordOfferV2.minNumberInRotateArray(array);
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
            return Integer.MIN_VALUE;
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


}
