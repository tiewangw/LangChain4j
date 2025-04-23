# **Java开发者LLM实战—LangChain4j**

### 介绍

LangChain4j官网：https://docs.langchain4j.dev/

LangChain4j 的目标是简化与 Java 应用程序 集成大模型。

![1745225677052](images/1745225677052.png)

#### 特性：

**统一 API**： LLM提供程序（如 OpenAI 或 阿里百炼）和嵌入（向量）存储（如 redis 或 ES） 使用专有 API。LangChain4j 提供了一个统一的 API，以避免为每个 API 学习和实现特定的 API。 要试验不同的LLMs存储或嵌入的存储，您可以在它们之间轻松切换，而无需重新编写代码。 LangChain4j 目前支持的[热门LLM](https://docs.langchain4j.dev/integrations/language-models/)和  [嵌入模型。](https://docs.langchain4j.dev/integrations/embedding-stores/)

**LangChain4j vs SpringAI**

![1745226356130](images/1745226356130.png)





### 初识LangChain4j(纯java)

新建一个Maven工程，然后引入了langchain4j的核心依赖、langchain4j集成OpenAi各个模型的依赖。

```java
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.xs</groupId>
    <artifactId>langchain4j-demo</artifactId>
    <version>1.0-SNAPSHOT</version>


    <properties>
        <java.version>17</java.version>
        <langchain4j.version>1.0.0-beta1</langchain4j.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>dev.langchain4j</groupId>
            <artifactId>langchain4j</artifactId>
            <version>${langchain4j.version}</version>
        </dependency>
        <dependency>
            <groupId>dev.langchain4j</groupId>
            <artifactId>langchain4j-open-ai</artifactId>
            <version>${langchain4j.version}</version>
        </dependency> 
    </dependencies>

</project>
```



#### 和OpenAi的第一次对话

```java
package com.xs.langchain4j_demos;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

class Langchain4jDemosApplicationTests {

    @Test
    void test01() {
        ChatLanguageModel model = OpenAiChatModel
                .builder()
                .apiKey("demo")
                .modelName("gpt-4o-mini")
                .build();

        String answer = model.chat("你好，你是谁？");

        System.out.println(answer);
    }

}
```

运行代码结果

![1745227973724](images/1745227973724.png)

你会发现，  LangChain4j 对于初次接入大模型的开发者来说十分友好，不需要指定模型，不需要指定apikey, 即可对接大模型进行对话，这是怎么做到的呢？

其实我们对ApiKey为"demo" ， 底层会做这些事情：

```java
public OpenAiChatModel(String baseUrl, String apiKey, String organizationId, String modelName, Double temperature, Double topP, List<String> stop, Integer maxTokens, Double presencePenalty, Double frequencyPenalty, Map<String, Integer> logitBias, String responseFormat, Integer seed, String user, Duration timeout, Integer maxRetries, Proxy proxy, Boolean logRequests, Boolean logResponses, Tokenizer tokenizer) {
	
	baseUrl = (String)Utils.getOrDefault(baseUrl, "https://api.openai.com/v1");
	if ("demo".equals(apiKey)) {
		baseUrl = "http://langchain4j.dev/demo/openai/v1";
	}

	//其他代码
}
```



