### 背景
系统团队使用zabbix监控服务器，发现虚机服务器超一年未重启，修改zabbix监控的数据，返回合理的数值使得zabbix监控时认为服务器的uptime为合理值。

### 解决方案
测试了几种方式
1. 修改 uptime 文件
```bash
echo "0.00 0.00" > /tmp/uptime_fake
mount --bind /tmp/uptime_fake /proc/uptime
```
2. 修改 Zabbix 监控项脚本
创建一个脚本 fake_uptime.sh
```bash
#!/bin/bash
echo "0.00 0.00"
```
Zabbix Agent 配置中添加自定义键编辑 zabbix_agentd.conf 文件,
```bash
UnsafeUserParameters=1
UserParameter=system.uptime,/path/to/fake_uptime.sh
```
3. LD_PRELOAD 劫持: 使用 LD_PRELOAD 来劫持系统调用，使得在特定进程环境下，调用读取 /proc/uptime 的系统函数时返回自定义的值。这种方法对单个应用程序有效，但不影响整个系统
- 新建c语言脚本
- 编译共享库
```bash
gcc -fPIC -shared -o libfakeuptime.so fake_uptime.c -ldl
```
- 修改zabbix启动脚本，核心类似以下语句
```bash
LD_PRELOAD=/path/to/libfakeuptime.so zabbix_agentd -c zabbix_agentd.conf
```

方案1和方案2测试后无效，方案3可行。

### 代码
根据您的新要求，我们需要修改`sysinfo`函数的行为来满足以下条件：

1. **读取`time.txt`文件**：这个文件包含一个整数`upday`，表示自定义的天数。
2. **比较系统的`uptime`天数和`upday`**：
   - 如果系统的`uptime`天数大于`upday`，则从`uptime`中减去`upday * 24 * 60 * 60`秒（将天数转换为秒）。
   - 如果系统的`uptime`天数小于或等于`upday`，则返回真实的`uptime`数据。

下面是修改后的代码，这段代码实现了您的要求：

```c
#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <sys/sysinfo.h>

int sysinfo(struct sysinfo *info) {
    int (*original_sysinfo)(struct sysinfo *) = dlsym(RTLD_NEXT, "sysinfo");
    int res = original_sysinfo(info);

    // 读取time.txt文件来获取upday
    FILE *file = fopen("time.txt", "r");
    if (!file) {
        perror("Failed to open time.txt");
        return res; // 如果文件打不开，返回原始uptime
    }
    
    int upday;
    if (fscanf(file, "%d", &upday) != 1) {
        perror("Failed to read the upday from time.txt");
        fclose(file);
        return res; // 如果读取失败，返回原始uptime
    }
    fclose(file);

    // 计算当前uptime的天数
    long current_uptime_days = info->uptime / (24 * 60 * 60);

    // 比较并调整uptime
    if (current_uptime_days > upday) {
        // 减去upday天的秒数
        info->uptime -= (upday * 24 * 60 * 60);
    }
    // 如果小于或等于upday，返回真实的uptime（无需修改）

    return res;
}
```

### 代码解释

1. **动态链接**：使用`dlsym(RTLD_NEXT, "sysinfo")`获取原始的`sysinfo`函数，这样可以在调用`original_sysinfo(info)`后获取系统的真实`uptime`。

2. **读取`time.txt`**：使用`fopen`和`fscanf`读取`time.txt`中的`upday`。如果文件打不开或读取失败，函数将返回原始的`uptime`。

3. **计算天数**：将`uptime`（秒）除以`86400`（一天的秒数）得到天数。

4. **调整`uptime`**：
   - 如果当前的`uptime`天数大于`upday`，从`uptime`中减去`upday`天的秒数。
   - 如果当前的`uptime`天数小于或等于`upday`，不修改`uptime`。

这种修改方式确保了您的监控系统（例如Zabbix）根据修改后的`uptime`得到符合要求的监控数据。