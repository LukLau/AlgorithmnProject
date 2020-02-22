package org.dora.algorithm.swordoffer;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/2/19
 */
public class CharacterStream {

    //Insert one char from stringstream
    private int[] hash = new int[256];

    private int index = 1;

    public void Insert(char ch) {
        if (hash[ch] == 0) {
            hash[ch] = index;
        } else if (hash[ch] >= 1) {
            hash[ch] = -1;
        }
        index++;
    }

    //return the first appearence once char in current stringstream
    public char FirstAppearingOnce() {
        char ans = '#';
        int minIndex = Integer.MAX_VALUE;
        for (int i = 0; i < hash.length; i++) {
            if (hash[i] >= 1 && hash[i] < minIndex) {
                ans = (char) i;
                minIndex = hash[i];
            }
        }
        return ans;
    }


}
