package org.dora.algorithm.designpattern;

/**
 * @author liulu
 * @date 2019-04-14
 */
public class MultiThread {
    public static void main(String[] args) {
        Wrapper wrapper = new Wrapper(1);
        OddThread oddThread = new OddThread(wrapper);
        EvenThread evenThread = new EvenThread(wrapper);

        evenThread.start();
        oddThread.start();
    }
}


class Wrapper {
    public int nums;
    public boolean odd = true;

    public Wrapper(int nums) {
        this.nums = nums;
    }
}

class OddThread extends Thread {

    private Wrapper wrapper;

    public OddThread(Wrapper wrapper) {
        this.wrapper = wrapper;
    }

    /**
     * When an object implementing interface <code>Runnable</code> is used
     * to create a thread, starting the thread causes the object's
     * <code>run</code> method to be called in that separately executing
     * thread.
     * <p>
     * The general contract of the method <code>run</code> is that it may
     * take any action whatsoever.
     *
     * @see Thread#run()
     */
    @Override
    public void run() {
        while (wrapper.nums < 10) {
            synchronized (wrapper) {
                if (wrapper.odd) {
                    System.out.println("奇数" + wrapper.nums + "->" + Thread.currentThread().getName());
                    wrapper.nums++;
                    wrapper.odd = false;
                    wrapper.notifyAll();
                    ;
                } else {
                    try {
                        wrapper.wait();
                    } catch (Exception e) {

                    }
                }

            }
        }
    }
}

class EvenThread extends Thread {

    private Wrapper wrapper;

    public EvenThread(Wrapper wrapper) {
        this.wrapper = wrapper;
    }

    /**
     * When an object implementing interface <code>Runnable</code> is used
     * to create a thread, starting the thread causes the object's
     * <code>run</code> method to be called in that separately executing
     * thread.
     * <p>
     * The general contract of the method <code>run</code> is that it may
     * take any action whatsoever.
     *
     * @see Thread#run()
     */
    @Override
    public void run() {
        while (wrapper.nums < 10) {
            synchronized (wrapper) {
                if (!wrapper.odd) {
                    System.out.println("偶数" + wrapper.nums + "->" + Thread.currentThread().getName());
                    wrapper.nums++;
                    wrapper.odd = true;
                    wrapper.notifyAll();
                } else {
                    try {
                        wrapper.wait();
                    } catch (Exception e) {

                    }
                }

            }
        }
    }
}

