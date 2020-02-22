package org.dora.algorithm.swordoffer;

import org.dora.algorithm.datastructe.TreeNode;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/2/21
 */
public class SerializeTree {

    private int index = -1;

    public String Serialize(TreeNode root) {
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

    public TreeNode Deserialize(String str) {
        if (str == null || str.isEmpty()) {
            return null;
        }
        String[] words = str.split(",");
        return intervalDeserialize(words);
    }

    private TreeNode intervalDeserialize(String[] words) {
        index++;
        if (index >= words.length) {
            return null;
        }
        String word = words[index];
        if ("#".equals(word)) {
            return null;
        } else {
            TreeNode node = new TreeNode(Integer.parseInt(word));
            node.left = intervalDeserialize(words);
            node.right = intervalDeserialize(words);
            return node;
        }
    }

}
