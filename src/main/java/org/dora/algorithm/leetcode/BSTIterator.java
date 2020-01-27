package org.dora.algorithm.leetcode;

import org.dora.algorithm.datastructe.TreeNode;

import java.util.LinkedList;
import java.util.Stack;

/**
 * @author liulu12@xiaomi.com
 * @date 2019/12/4
 */
public class BSTIterator {


    private LinkedList<Integer> ans;

    public BSTIterator(TreeNode root) {
        ans = new LinkedList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            ans.add(p.val);
            p = p.right;
        }
    }

    /**
     * @return the next smallest number
     */
    public int next() {
        return hasNext() ? ans.poll() : -1;
    }

    /**
     * @return whether we have a next smallest number
     */
    public boolean hasNext() {
        return !ans.isEmpty();
    }
}
