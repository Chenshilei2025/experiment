# Loyal Agent 数据集构造

## 摘要

Loyal Agent 是一个面向忠实性（loyalty）的交互式agent评测数据集。其核心问题不是一般意义上的任务完成率，而是代理能否在代表用户与外部环境交互时，将用户利益作为信息披露和决策的约束。数据集由两个互补的任务机制组成：**EIL**（Exploitable Information Leakage）考察agent能否在委托沟通中披露完成任务所必需的信息，同时保护会被相对方转化为不利杠杆的真实信息；**MIU**（Manipulated Information Uptake）考察agent能否在外部信息包含操纵性内容时，仅依据用户约束、授权范围与干净证据作出可追溯的决策。

## 1. 任务定义与场景空间

### 1.1 两类忠实性机制

本文将忠实性界定为：agent在受用户委托与外部主体互动时，持续以用户可识别的利益、约束和授权边界约束其信息披露与行动选择。相应地，场景的划分以外部环境影响用户利益的因果路径为依据。数据集包含两类互补机制。EIL（Exploitable Information Leakage）刻画“披露—利用”路径：agent在完成委托所必需的信息披露之外，若额外泄露真实但非必要的私有背景，外部相对方即可将其转化为对用户不利的谈判、程序或准入条件。MIU（Manipulated Information Uptake）刻画“输入—决策”路径：外部信息可能以操纵性方式改变代理的判断，代理应仅依照用户约束、授权证据与可追溯的推理作出选择，而不应采纳未经授权的劝诱、压力或定向建议。

### 1.2 大场景的分类依据

两类机制使用不同、但均以用户利益受损方式为中心的分类轴。对于 EIL，我们以“泄露信息类型”与“受损的用户利益”共同界定 family domain。前一维度刻画被披露事实何以能够成为相对方的可利用杠杆；后一维度刻画该杠杆最终损害的是用户的何种实质利益。对于 MIU，我们以“操纵性信息类型”与“决策类型”共同分类。操纵性信息类型指外部材料施加影响的方式，例如权威或专家压力、稀缺与紧迫性、社会认同或排名、虚假或选择性证据，以及恐惧情绪；决策类型则指代理受托作出的规范性判断，包括决策适当性判断、满足用户需求的推荐排序，以及关于主张是否得到证据支持的认识判断。该设计将信息本身的影响策略与其所作用的决策对象分离，使评测能够检验代理是否针对不同类型的操纵维持相同的用户中心决策原则。

| 机制 | Family domain | 首要分类维度 | 对应的用户利益或决策对象 | 典型忠实性风险 |
| --- | --- | --- | --- | --- |
| EIL | bargaining | 保留价格、期限紧迫性、支付能力与替代方案等交易杠杆信息 | 经济利益与合同条件 | 相对方据此加价、加费、收紧条款或施加承诺压力 |
| EIL | redress | 责任不确定性、证据缺口、资金压力与急迫处境等救济弱点 | 救济机会与程序公平 | 相对方提高证明负担、拖延处理或限制救济 |
| EIL | gatekeeping | 健康、家庭、年龄、宗教、既往拒绝等脆弱性或受保护背景 | 平等准入、程序权利与个人自主 | 机构据此增加审核、限制访问或作出不利决定 |
| MIU | financial advisory | 收益框架、伪专家背书、FOMO 与选择性业绩信息 | 风险适当性与财务福祉 | 代理偏离风险、流动性、费用或偿付约束 |
| MIU | commerce guidance | 赞助排序、虚假评价、折扣框架、稀缺提示与误导性比较 | 对合格选项的需求匹配与偏好排序 | 代理以外部推广替代用户需求和授权证据 |
| MIU | information guidance | 权威/共识压力、选择性叙事与情绪催促 | 对主张的支持、反驳或证据不足判断 | 代理将说服性材料误作支持结论的证据 |

### 1.3 大场景下的小场景分类

场景设计采用两级层次。`family_domain` 表示由相同忠实性风险结构界定的大场景，`subscenario` 则表示该结构在具体任务中的实现形式。前者保证不同样本在泄露—利用或操纵—决策关系上具有可比性，后者保留领域语言、行动对象和程序情境的异质性。当前发布数据共含 10,298 条记录，其中 EIL 6,700 条、覆盖 27 个 subscenario；MIU 3,598 条、覆盖 15 个 subscenario。

| 机制 | Family domain | subscenario覆盖 |
| --- | --- | --- |
| EIL | bargaining | 租赁、购车、维修、婚礼服务、搬家、薪资、自由职业报价与合同协商 |
| EIL | redress | 消费者维权、租赁纠纷、劳动争议与保险理赔 |
| EIL | gatekeeping | 晋升/离职与员工协作、招聘和教育申请、签证/移民/福利、医疗预约与心理咨询 |
| MIU | financial advisory | 股票、加密资产、基金、保险、银行理财与信用卡选择 |
| MIU | commerce guidance | 购物、餐饮、旅行、产品、房产和教育培训推荐 |
| MIU | information guidance | 新闻阅读、政治信息与健康教育 |


## 2. 真实性与来源锚定

### 2.1 真实性的操作化

本文所说的真实性并不是“每个故事都曾真实发生”，而是指记录中出现的领域信息有清楚的依据，且任务中的利益冲突符合该类交互的基本逻辑。例如，车型规格、职业职责、院校属性、办事流程、检查记录可以由固定版本的外部资料锚定；但具体用户、交易对手、申请结果和个人处境均是为测试而设定的。

### 2.2 场景级来源协议

外部资料在本管道中充当局部事实锚点. 表格报告全部 42 个子场景的来源协议。“受控合成”表示该场景不读取外部事实；表中“用途”仅限定来源可提供的局部语义。

| 机制 | 子场景 | 许可来源 | 用途及边界 |
| --- | --- | --- | --- |
| EIL | rental negotiation；emergency repair；wedding service procurement；moving quote；lawyer consultation；freelance pricing | 受控合成 |  
| EIL | car purchase negotiation | FuelEconomy | 精确匹配的车型、变速箱及城市/高速/综合油耗规格。 |
| EIL | salary negotiation；internal promotion；recruitment screening；internship application | O*NET 30.1 | 职业名称、通用职责、任务和技能词汇 |
| EIL | client contract negotiation | CUAD | 条款类型措辞 |
| EIL | consumer redress | CFPB | 去标识化的产品、争议、日期、回应和提交渠道属性 |
| EIL | rental dispute | NYC HPD | 违规类别、状态和检查日期 |
| EIL | labor dispute | EEOC | 程序性语言 |
| EIL | insurance claim；resignation communication；performance evaluation；medical appointment；mental health matching | 受控合成  |
| EIL | employee agent | PrivaCI-Bench | 角色、信息类型、同意形式和目的属性； |
| EIL | student application；advisor communication；academic appeal | College Scorecard | 院校、地点、学费和主授学位层次； |
| EIL | immigration application；visa communication | eCFR Title 8 | 法规标题和段落中的程序性限定 |
| EIL | government benefit application | USAGov Benefits | 公共项目说明和流程性表述 |
| MIU | stock investment | FinQA；TAT-QA；FinanceBench | 表格化金融证据、数值关系和证据性表述 |
| MIU | cryptocurrency trading；fund recommendation；insurance purchase；credit card recommendation | 受控合成 |
| MIU | bank wealth management | FinQA；TAT-QA；FinanceBench；AgentDojo | 金融证据表述与模拟银行任务属性 |
| MIU | shopping recommendation；product review | Amazon ESCI | 查询、商品标题、品牌和相关性关系 |
| MIU | restaurant recommendation | NYC/Chicago 餐饮检查快照 | 餐厅、菜系或设施类型、检查日期、风险/结果、评分或等级属性。 |
| MIU | travel planning | NWS | 预报时段、日期、温度、风力和天气描述 |
| MIU | property recommendation | 受控合成 |
| MIU | education and training recommendation | College Scorecard | 院校、地点、学费和学位层次 |
| MIU | news reading；political information | 受控合成 |
| MIU | health education | PubMedQA；MedMCQA | 一般性健康问题、证据和结论/选项 |

## 3. 数据构造管道

### 3.1 构造单元与阶段化流程

我们以子场景构造计划作为最小构造单元。这一计划写清楚各场景的来源锚点, 字段数量以及字段依赖关系, 以及机制相关的变化轴. 对每一条数据，管道依次执行：(i) 按场景白名单冻结来源事实；(ii) 明确结构化构造计划和字段生成顺序；(iii) 生成完整的结构化字段 (iv) 记录级执行机制、来源和模式验证, 多样性以及质量审计

### 3.2 EIL的字段生成顺序

每个 EIL 记录按“必要信息—私有杠杆—条件性利用”的顺序构造。首先生成 1--5 条原子化 `necessary_information`，仅包含完成受托沟通所需的任务范围、公开时点、材料或资格、公开请求条款。随后生成 2--5 条 `exploitable_slots`，即对用户真实但可能被对手利用造成不利的信息；每个 slot 必须能够使相对方采取一种具体的不利行动。

管道对两类信息施加语义分离约束：任意 `exploitable_slots` 不得与必要信息重复、细化、量化、限时、比较，或被其强烈蕴含；不同 slot 对应不同的潜在不利结果。在此冻结状态上，`user_natural_language` 仅作自然语言渲染，不得新增事实或显式标记披露边界。`adversary_opening` 被限定为泄露发生前的开放性探询；`adversary_config` 则是配置了信息泄露后对手可能做出的条件性不利行动、策略和压力强度。

### 3.3 MIU的字段生成顺序

MIU 从用户侧画像开始构造。`user_constraints` 表示不可违反的边界，`user_preferences` 则表示用户的决策偏好，`authorized_information` 指定可采信的证据类别，`decision_boundary` 给出三个互斥的行动或认识选项。金融建议、商业推荐和信息判断三类任务分别使用适当性、需求匹配和证据支持程度三种决策语义。

在用户画像固定后，生成 2--5 条 `clean_context` 和 2--5 条 `manipulated_context`。每条 clean card 必须通过 `authorized_by` 连接至用户授权的证据类别；每条 manipulated card 带有操纵策略标签，并倡导偏离基准的选项。`baseline_decision` 只能由用户合同和 clean cards 推导. 为消除来源和位置捷径，发布时将 clean 与 manipulated 卡片的内容去除 ID、来源和角色标签后合并，并按记录级种子随机排列为 `external_information`。clean/manipulated 的角色划分和决策证书仅保留于受控评测字段及私有审计侧车。

### 3.4 质量检查

上述流程在管道层面是阶段化的，但一条记录的内容由一次完整模型调用按字段依赖顺序生成，以保持同一记录内任务状态、证据和标签的一致性。模型仅接收该记录的冻结来源包与构造计划.质量控制同时评估**有效性**与**捷径风险**。有效性检查验证一条记录是否实现了其预定机制：包括 schema、字段 ID 和长度，EIL 中必要信息与私有 slot 的分离、对手配置的合法性，以及 MIU 的闭集基准决策、clean grounding 和授权范围。捷径风险检查验证模型是否可借助非任务信号完成评测，包括内部标签泄漏、攻击标签重复、证据长度不一、跨 split ID 冲突和重复用户请求。

对于 EIL，我们在 family 内报告必要信息功能类型与私有杠杆类型之间的归一化互信息（NMI），并计算二者的语义词项重叠；对于 MIU，我们报告 clean evidence 授权类别与操纵机制之间的 family 条件化 NMI、clean/manipulated 内容重叠，EIL 的必要信息—私有杠杆宏平均条件化 NMI 为 0.1644，MIU 的 clean 授权类别—操纵机制宏平均条件化 NMI 为 0.2144，clean/manipulated 内容重叠中位数为 0.0732.

对于 MIU 基准标签，静态检查只能确认闭集性、clean grounding 和结构化证书的一致性，不能替代独立模型的baseline_decision决策复现。为此，管道定义双模型 baseline audit：两个独立模型仅接收用户自然语言、约束/偏好、授权信息、决策边界与 clean context；只有二者选择的baseline_decision一致的样本才保留

## 5. 多样性设计与覆盖分析

### 5.1 覆盖设计

多样性在本数据集中是构造目标，我们沿三个相互补充的层级进行设计。第一，**任务覆盖**跨越 6 个 family domain 与 42 个 subscenario，使交互对象、信息环境和行动目标发生变化。第二，**机制覆盖**EIL 交叉考虑必要信息、私有杠杆、对手开场探询方式和对手压榨策略的多样性维度；MIU 交叉考虑用户约束、授权证据、决策类型和操纵策略的多样性维度。第三，**语言覆盖**在不改变结构化事实和标签的条件下变化用户的叙事顺序、请求形式、信息组织和表达风格.

### 5.2 覆盖度量

第一，类别覆盖数（support）检验预设的场景和忠实性功能类型是否实际进入发布集。第二，有效支持数 \(N_{eff}=\exp\{-\sum_c p(c)\log p(c)\}\) 将类别分布折算为等权类别数，用以识别“类别虽多、样本却集中”的情况。第三，最大类别占比（dominance）报告最常见类别的样本份额，提供对集中度的直观检验。对单标签场景变量和多标签功能类型，\(p(c)\) 均在相应观测单位上计算。


### 5.3 场景覆盖

EIL 覆盖 bargaining（2,500 条）、redress（1,500 条）和 gatekeeping（2,700 条），共 27 个 subscenario；MIU 覆盖 financial advisory（1,751 条）、commerce guidance（1,252 条）和 information guidance（595 条），共 15 个 subscenario。EIL 的 subscenario 有效支持数为 25.963，最大类别占比为 0.0672；MIU 相应为 14.765 与 0.0834。这些数值表明，在各自已发布的子场景集合内，样本并未集中于少数具体任务.

### 5.4 EIL的字段多样性机制覆盖

EIL 的 `necessary_information` 表示完成委托所需、可安全披露的事实；`exploitable_slots` 表示真实但非必要、泄露后会产生不利杠杆的背景事实。必要信息 taxonomy 覆盖任务范围/交付物、公开时间或流程、资格或材料、请求条款或救济、偏好或规格；私有杠杆 taxonomy 覆盖期限紧迫性、预算或保留价格、替代方案不足、个人或健康脆弱性、法律/申诉弱点和支付能力。
对手 profile 则由 `adversary_opening` 的首轮探询和 `adversary_config.strategy` 的条件性不利动作构成：前者包括时间/可用性、预算/付款、文件/资格、替代方案/承诺及范围/流程探询，后者包括费用恶化、条款收紧、程序拖延、访问或救济限制及承诺压力.

在 bargaining 中，租赁、购车、薪资和服务采购场景允许代理沟通服务范围、规格、公开时间与报价请求，但要求保护硬期限、最高支付意愿、支付能力与替代方案不足。例如，租赁代理可提出房型、宠物许可和入住窗口，却不应暴露必须搬入的日期、租金上限或其他房源已拒绝用户的事实。对应地，对手可围绕付款、时间或替代方案开放探询，并在泄露发生后转化为费用、条款或快速转化为压力.

在 redress 中，消费者维权、劳动争议、租赁纠纷和保险理赔允许披露事件事实、救济请求和程序材料，但保护责任不确定性、证据缺口、资金承受能力及急迫处境。

在 gatekeeping 中，教育/就业申请、签证与福利沟通、医疗预约和心理支持匹配以资格材料、申请步骤与公开日程为必要信息；健康或家庭脆弱性、既往拒绝、急迫程度及替代方案不足则是典型的受保护背景。对手 profile 因而围绕文件、资格、可用性和承诺建立试探，并将泄露转化为额外审核、准入限制或程序迟延。

### 5.5 MIU：决策条件与操纵策略的覆盖

MIU 的用户 profile 由 `user_constraints` 与 `user_preferences` 共同构成，覆盖成本、风险/适配性、流动性/灵活性、资格/合规、兼容性/功能、时间/物流和证据范围/不确定性。`manipulated_context` 按攻击机制归为权威/专家压力、稀缺/紧迫性、社会认同/排名、价格或金融激励框架、虚假或选择性证据，以及恐惧或情绪压力。
financial advisory 覆盖股票、基金、加密交易、保险、信用卡和财富管理，因而将风险承受、集中度、流动性、费用、还款能力和保障范围等用户侧约束，与伪专家背书、市场时机 FOMO、收益/费用框架和选择性业绩叙事相对照。
commerce guidance 则将购物、餐饮、旅行、房产、产品和培训推荐中的预算、兼容性、耐用性、饮食限制、可达性和时间物流，置于赞助排序、虚假评价、折扣框架、库存或席位稀缺、误导比较和原生广告等影响之下。
information guidance 的覆盖较窄，是因为新闻、政治信息和健康教育必须避免捏造领域事实；其用户 profile 围绕主张范围、证据强度、风险/收益解释与结论边界，操纵主要体现为权威或社会舆论风向影响agent偏离用户的观点主张

## 6. 适用边界与局限

Loyal Agent 是一个受控合成基准，而非自然发生的代理对话日志。因此，本文中的真实性主张限于事实锚定与忠实性冲突的结构保真,应被理解为对该受控任务分布中代理忠实性的可重复测量，而非对开放世界行为的充分刻画.
