package org.dora.algorithm.geeksforgeek;

import java.util.HashMap;
import java.util.Map;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/1/20
 */
public class ValidWordAbbr {
    private Map<String, Integer> dict = new HashMap<>();
    private Map<String, Integer> abbr = new HashMap<>();

    /**
     * @param dictionary: a list of words
     */
    public ValidWordAbbr(String[] dictionary) {
        // do intialization if necessary
        for (String s : dictionary) {
            dict.put(s, dict.getOrDefault(s, 0) + 1);
            String a = constructItem(s);
            abbr.put(a, abbr.getOrDefault(a, 0) + 1);
        }

    }

    public static void main(String[] args) {
        String[] strings = new String[]{"deer", "door", "cake", "card"};
        ValidWordAbbr validWordAbbr = new ValidWordAbbr(strings);
        validWordAbbr.isUnique("a");
    }

    /**
     * @param word: a string
     * @return: true if its abbreviation is unique or false
     */
    public boolean isUnique(String word) {
        // write your code here
        String a = constructItem(word);
        return dict.getOrDefault(word, 0).equals(abbr.getOrDefault(a, 0));
    }

    private String constructItem(String str) {
        int len = str.length();
        if (len <= 2) {
            return str;
        }
        StringBuilder builder = new StringBuilder();
        builder.append(str.charAt(0));
        int count = len - 2;
        builder.append(count);
        builder.append(str.charAt(len - 1));
        return builder.toString();
    }

}
