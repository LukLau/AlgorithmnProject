package org.dora.algorithm.leetcode.v3;

import org.dora.algorithm.datastructe.ListNode;
import org.dora.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/4/6
 */
public class LeetCodeP1 {

    public static void main(String[] args) {
        ListNode head = new ListNode(-10);

        head.next = new ListNode(-3);

        head.next.next = new ListNode(0);

        head.next.next.next = new ListNode(5);

        head.next.next.next.next = new ListNode(9);

        LeetCodeP1 p1 = new LeetCodeP1();
        p1.sortedListToBST(head);
    }

    /**
     * 100. Same Tree
     *
     * @param p
     * @param q
     * @return
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        if (p.val != q.val) {
            return false;
        }
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);

    }

    /**
     * 101. Symmetric Tree
     *
     * @param root
     * @return
     */
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isSymmetric(root.left, root.right);
    }

    private boolean isSymmetric(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        if (left.val != right.val) {
            return false;
        }
        return isSymmetric(left.left, right.right) && isSymmetric(right.left, left.right);
    }


    /**
     * 102. Binary Tree Level Order Traversal
     *
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        LinkedList<TreeNode> list = new LinkedList<>();
        list.add(root);
        List<List<Integer>> result = new ArrayList<>();
        while (!list.isEmpty()) {
            int size = list.size();
            List<Integer> tmp = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode poll = list.poll();
                tmp.add(poll.val);
                if (poll.left != null) {
                    list.offer(poll.left);
                }
                if (poll.right != null) {
                    list.offer(poll.right);
                }
            }
            result.add(tmp);
        }
        return result;
    }

    /**
     * 103. Binary Tree Zigzag Level Order Traversal
     *
     * @param root
     * @return
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        LinkedList<TreeNode> deque = new LinkedList<>();
        deque.offer(root);
        boolean leftToRight = true;
        while (!deque.isEmpty()) {
            int size = deque.size();

            int[] tmp = new int[size];

            for (int i = 0; i < size; i++) {

                TreeNode poll = deque.poll();

                int index = leftToRight ? i : size - 1 - i;

                tmp[index] = poll.val;

                if (poll.left != null) {
                    deque.offer(poll.left);
                }
                if (poll.right != null) {
                    deque.offer(poll.right);
                }
            }
            List<Integer> list = new ArrayList<>();
            for (int num : tmp) {
                list.add(num);
            }
            ans.add(list);
            leftToRight = !leftToRight;
        }
        return ans;
    }

    /**
     * 104. Maximum Depth of Binary Tree
     *
     * @param root
     * @return
     */
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }


    public int maxDepthII(TreeNode root) {
        if (root == null) {
            return 0;
        }
        HashMap<TreeNode, Integer> hashMap = new HashMap<>();
        return maxDepthII(hashMap, root);
    }

    private int maxDepthII(HashMap<TreeNode, Integer> map, TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (map.containsKey(root)) {
            return map.get(root);
        }
        map.put(root, 1);
        return 1 + Math.max(maxDepthII(map, root.left), maxDepthII(map, root.right));
    }


    /**
     * 105. Construct Binary Tree from Preorder and Inorder Traversal
     *
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder == null || inorder == null) {
            return null;
        }
        return intervalBuildTree(0, inorder.length - 1, inorder, 0, preorder);
    }

    private TreeNode intervalBuildTree(int inStart, int inEnd, int[] inorder, int preStart, int[] preorder) {
        if (inStart > inEnd || preStart >= preorder.length) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[preStart]);

        int i = inStart;
        while (i <= inEnd) {
            if (inorder[i] == root.val) {
                break;
            }
            i = i + 1;
        }
        root.left = intervalBuildTree(inStart, i - 1, inorder, preStart + 1, preorder);
        root.right = intervalBuildTree(inStart + i - inStart + 1, inEnd, inorder, preStart + i - inStart + 1, preorder);
        return root;
    }


    /**
     * 106. Construct Binary Tree from Inorder and Postorder Traversal
     *
     * @param inorder
     * @param postorder
     * @return
     */
    public TreeNode buildTreeV2(int[] inorder, int[] postorder) {
        if (inorder == null || postorder == null) {
            return null;
        }
        return intervalBuildTreeV2(0, inorder.length - 1, inorder, 0, postorder.length - 1, postorder);
    }

    private TreeNode intervalBuildTreeV2(int inStart, int inEnd, int[] inorder, int postStart, int postEnd, int[] postorder) {
        if (inStart > inEnd || postStart > postEnd) {
            return null;
        }
        TreeNode root = new TreeNode(postorder[postEnd]);

        int i = inStart;
        while (i <= inEnd) {
            if (inorder[i] == root.val) {
                break;
            }
            i++;
        }
        root.left = intervalBuildTreeV2(inStart, i - 1, inorder, postStart, postStart + i - inStart - 1, postorder);

        root.right = intervalBuildTreeV2(i + 1, inEnd, inorder, postStart + i - inStart, postEnd - 1, postorder);

        return root;
    }

    /**
     * 108. Convert Sorted Array to Binary Search Tree
     *
     * @param nums
     * @return
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums == null || nums.length == 0) {
            return null;
        }
        return sortedArrayToBST(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBST(int[] nums, int start, int end) {
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBST(nums, start, mid - 1);
        root.right = sortedArrayToBST(nums, mid + 1, end);
        return root;
    }

    /**
     * todo
     * 109. Convert Sorted List to Binary Search Tree
     *
     * @param head
     * @return
     */
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) {
            return null;
        }
        return sortedListToBST(head, null);
    }

    private TreeNode sortedListToBST(ListNode start, ListNode end) {
        if (start == end) {
            return null;
        }
        ListNode fast = start;
        ListNode slow = start;
        while (fast != end && fast.next != end) {
            fast = fast.next.next;
            slow = slow.next;
        }
        TreeNode root = new TreeNode(slow.val);
        root.left = sortedListToBST(start, slow);
        root.right = sortedListToBST(start.next, end);
        return root;
    }


    /**
     * 110. Balanced Binary Tree
     *
     * @param root
     * @return
     */
    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        }
        int left = maxDepthII(root.left);
        int right = maxDepthII(root.right);
        if (Math.abs(left - right) <= 1) {
            return isBalanced(root.left) && isBalanced(root.right);
        }
        return false;
    }


    /**
     * 111. Minimum Depth of Binary Tree
     *
     * @param root
     * @return
     */
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return 1;
        }
        if (root.left == null) {
            return 1 + minDepth(root.right);
        }
        if (root.right == null) {
            return 1 + minDepth(root.left);
        }
        return 1 + Math.min(minDepth(root.left), minDepth(root.right));
    }


    /**
     * 112. Path Sum
     *
     * @param root
     * @param sum
     * @return
     */
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }

        if (root.left == null && root.right == null && root.val == sum) {
            return true;
        }
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }


    /**
     * 113. Path Sum II
     *
     * @param root
     * @param sum
     * @return
     */
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> ans = new ArrayList<>();
        pathSum(ans, new ArrayList<>(), root, sum);
        return ans;
    }

    private void pathSum(List<List<Integer>> ans, List<Integer> tmp, TreeNode root, int sum) {
        tmp.add(root.val);
        if (root.left == null && root.right == null && root.val == sum) {
            ans.add(new ArrayList<>(tmp));
        } else {
            if (root.left != null) {
                pathSum(ans, tmp, root.left, sum - root.val);
            }
            if (root.right != null) {
                pathSum(ans, tmp, root.right, sum - root.val);
            }
        }
        tmp.remove(tmp.size() - 1);
    }


    /**
     * 114. Flatten Binary Tree to Linked List
     *
     * @param root
     */
    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        stack.push(p);
        TreeNode prev = null;
        while (!stack.isEmpty()) {
            TreeNode pop = stack.pop();
            if (prev != null) {
                prev.left = null;
                prev.right = pop;
            }
            if (pop.right != null) {
                stack.push(pop.right);
            }
            if (pop.left != null) {
                stack.push(pop.left);
            }
            prev = pop;
        }
    }


    /**
     * 115. Distinct Subsequences
     *
     * @param s
     * @param t
     * @return
     */
    public int numDistinct(String s, String t) {

    }


}
