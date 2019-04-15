package org.dora.algorithm.solution;

import org.dora.algorithm.datastructe.TreeNode;

import java.util.Arrays;
import java.util.Deque;
import java.util.LinkedList;

/**
 * @author liulu
 * @date 2019-04-14
 */
public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder stringBuilder = new StringBuilder();
        buildString(stringBuilder, root);
        return stringBuilder.toString();
    }

    private void buildString(StringBuilder stringBuilder, TreeNode root) {
        if (root == null) {
            stringBuilder.append("#,");
            return;
        }
        stringBuilder.append(root.val).append(",");
        buildString(stringBuilder, root.left);
        buildString(stringBuilder, root.right);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Deque<String> nodes = new LinkedList<>();
        nodes.addAll(Arrays.asList(data.split(",")));
        return buildTreeNode(nodes);
    }

    private TreeNode buildTreeNode(Deque<String> nodes) {
        String val = nodes.remove();
        if (val.equals("#")) {
            return null;
        } else {
            TreeNode node = new TreeNode(Integer.parseInt(val));
            node.left = buildTreeNode(nodes);
            node.right = buildTreeNode(nodes);
            return node;
        }
    }
}
