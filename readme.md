

昨天 只是造轮子失败了。那今天就不造轮子。而没必要全盘放弃。

今天，我们不从底层写算法。用别人的成熟 代码。

需求：解决至少一个gym游戏。

造轮子的项目叫diy
认真学习别人的项目叫learn

本项目内容是强化学习rl
所以项目叫 learn_rl


当务之急是找到一个能跑的代码。我已经不想再修改自己的了。大概是不会成功的。复制别人的项目，赶紧把功能实现。

要求1   能收敛
2   一定要是神经网络收敛的。（其它算法的以后再考虑，目前必须用神经网络（不然显卡不是白买了么。。现在流行神经网络，神经网络要先学））


代码我要看到三个部分。

游戏，强化学习，神经网络

运行后我要看到算法确实向有用的方向收敛。



https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow.git
刚才这个 ，能看到游戏，但看不到神经网络

真别造轮子了，好好看看别人的代码吧。
认真学点东西回来。




真的累。找到个好用的代码都难。
真不怪我喜欢 造轮子。妈的根本找不到好吧。


还是先学理论吧。实践现在走不动了。
造轮子失败，找不到教程，玩你妈个b

https://blog.csdn.net/qq_41871826/article/details/108263919


最气的是，最近 gym包改了内容，而网上的教程都有点旧。我需要调试tmd

1   state 由原来的np数组变成了个tuple，要取[0]
2   step加了一个值

原来是
s_, r, done, info = env.step(a)

现在是


https://blog.csdn.net/qq_41871826/article/details/108263919
他参考的莫烦

拷贝的代码报错了

https://blog.csdn.net/sinat_29957455/article/details/103487477


https://www.cnblogs.com/Renyi-Fan/p/13772136.html


不是自己写的代码真的漏洞百出。
报错太多了。


真的是个严重的问题。
你从网上拷贝代码，是可以，但一旦报错。就非常麻烦了。有时容易修改，有时非常难修改。

报错的原因，
1   引用的库byd更新了。不兼容。妈的不兼容是最烦人的。

我现在这个就很难 。
主要有两方面，我没懂他的细节原理
2 即使我来写，我不会用他那样的习惯。


这就是原因了。


现在我放弃。我专心看原理。

代码也是要下载的。但我们这样：
如果代码不能直接运行。
两条路1 不实现了，只看看原理就够
2   认真学习原理，重写一遍。

https://www.bilibili.com/video/BV1Vx411j7kT/?p=27&vd_source=88a8cff72324a68b12af164215c67b12
