package org.dora.algorithm.geeksforgeek;

import org.dora.algorithm.datastructe.TreeNode;

import java.util.Arrays;
import java.util.Deque;
import java.util.LinkedList;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/1/21
 */
public class CodecTree {

    private int index = -1;

    public static void main(String[] args) {
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        TreeNode right = new TreeNode(3);
        root.right = right;

        right.left = new TreeNode(4);
        right.right = new TreeNode(5);

        CodecTree codecTree = new CodecTree();
        String item = codecTree.serialize(root);
        System.out.println(item);
        TreeNode deserialize = codecTree.deserialize("1,2,#,#,3,4,#,#,5,#,#,");
        System.out.println(deserialize.toString());
    }

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder builder = new StringBuilder();
        intervalSerialize(builder, root);
        return builder.toString();
    }

    private void intervalSerialize(StringBuilder builder, TreeNode root) {
        if (root == null) {
            builder.append("#,");
            return;
        }
        builder.append(root.val + ",");
        intervalSerialize(builder, root.left);
        intervalSerialize(builder, root.right);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserializeV2(String data) {
        if (data == null || data.isEmpty()) {
            return null;
        }
        String[] split = data.split(",");
        Deque<String> nodes = new LinkedList<>(Arrays.asList(split));
        return intervalDeserialize(nodes);
    }

    private TreeNode intervalDeserialize(Deque<String> nodes) {
        String remove = nodes.remove();
        if (remove.equals("#")) {
            return null;
        } else {
            TreeNode root = new TreeNode(Integer.parseInt(remove));
            root.left = intervalDeserialize(nodes);
            root.right = intervalDeserialize(nodes);
            return root;
        }
    }

    public TreeNode deserialize(String data) {
        if (data == null || data.isEmpty()) {
            return null;
        }
        String[] split = data.split(",");
        return interval(split);

    }

    private TreeNode interval(String[] split) {
        index++;
        if (index >= split.length) {
            return null;
        }
        String item = split[index];
        if (item.equals("#")) {
            return null;
        } else {
            TreeNode root = new TreeNode(Integer.parseInt(split[index]));
            root.left = interval(split);
            root.right = interval(split);
            return root;
        }
    }


}
