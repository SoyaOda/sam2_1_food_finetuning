SAM2.1-Hiera-Large モデルへの移行修正プラン

1. チェックポイントファイル名と配置

SAM2.1-Hiera-Largeモデルの重みファイルは**sam2.1_hiera_large.ptです
GitHub
。Base+モデル（sam2.1_hiera_base_plus.pt）と同様に、クローンしたSAM2.1公式リポジトリ配下のexternal/sam2/checkpoints/ディレクトリに配置します（external/sam2は公式SAM2.1リポジトリのルート）
GitHub
。もし00_setup_env.sh実行時にdownload_ckpts.shを使用していない場合は、このsam2.1_hiera_large.ptを公式提供先からダウンロードして所定の場所に置いてください
GitHub
。ファイル名は正確に**sam2.1_hiera_large.ptとし、他のチェックポイントと同じフォルダに入れて管理します。

2. 設定ファイルの修正箇所（モデルチェックポイントとモデル設定）

現在使用している設定ファイル（例: configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yaml）では、Base+モデル用のチェックポイントとモデル構成ファイルが指定されています
GitHub
。具体的には以下の項目です。

model.checkpoint – 現在は./checkpoints/sam2.1_hiera_base_plus.ptとなっているので、これを**./checkpoints/sam2.1_hiera_large.pt**に変更します
GitHub
（※./checkpoints/はSAM2.1リポジトリ内を指しています）。

model.model_cfg – 現在はconfigs/sam2.1/sam2.1_hiera_b+.yaml（Base+モデルの構造定義）となっているので、これを**configs/sam2.1/sam2.1_hiera_l.yaml**に変更します
GitHub
。Largeモデル用の設定ファイル名はsam2.1_hiera_l.yamlです（SAM2.1公式リポジトリに含まれています）。

以上2箇所をLargeモデル対応の名前に差し替えることで、学習時にLargeモデルの重みと構造が正しく読み込まれます。なお、設定ファイル内の他のパラメータ（例えばimage_sizeやnum_maskmemなど）は、基本的にLargeモデルでも変更不要です。num_maskmemはプリトレイン済みモデルと一致している必要があり、デフォルトの7のままにします
GitHub
（Base+もLargeも7で一致）。これは、変更するとチェックポイントとの重み形状不整合エラーが発生するためです
GitHub
。

3. 他の影響を受けるパラメータと具体的変更提案

バッチサイズ: Largeモデルはパラメータ数が多くVRAM使用量が増えるため、バッチサイズの見直しが必要です。例えば、Base+モデルで32GB GPUに対しバッチサイズ2で学習していた場合、Largeモデルではバッチサイズ1に減らすことを検討してください
GitHub
。16GB程度のGPUでは、Base+でもバッチサイズ1かつメモリ最適化必須だったように
GitHub
、Largeではより厳しいメモリ管理（バッチサイズ1、場合によっては画像解像度ダウンやgradAccumの活用）が必要になる可能性があります。

 

学習率: 一般にモデルサイズが大きくなると勾配変動も大きくなる可能性がありますが、基本の初期学習率1e-4はそのままでも構いません
GitHub
。過学習や発散が見られる場合に限り1e-5程度まで下げるなど微調整してください（SAM2.1の推奨レンジは1e-5〜3e-4
GitHub
）。

 

最大オブジェクト数 (max_num_objects): Largeモデルでは1画像あたり複数オブジェクトのマスクを同時に処理する部分でメモリを多く消費するため、設定によってはこの上限数を下げることが有効です。現在Base+用設定では50（または20）程度になっている場合があります
GitHub
GitHub
。Largeモデルでメモリ逼迫が起きるようなら、例えば**max_num_objects: 25**程度に減らすことを提案します（メモリ最適化設定では実際に50から25へ削減しています
GitHub
）。極端な場合はさらに減らす（例: 10や5）ことで、メモリ使用量を大幅に下げられます
GitHub
GitHub
。

 

精度 (fp16/bf16): Largeモデルではfp32よりも半精度（特にA100ならbf16）を使う恩恵がさらに大きくなります。Base+でもA100 40GB環境ではbf16推奨でしたが
GitHub
、Largeでは可能な限りbf16で学習してください
GitHub
。VRAMに余裕がないGPU（例:T4など）ではfp16/bf16が難しい場合もありますが、その場合はバッチサイズ削減など他の対策で対応します。

 

その他、データローダーワーカー数やpin_memory設定については、Largeモデル固有というより全体のメモリ最適化の観点で検討します（既にWSL環境等ではワーカー数削減やpin_memory無効化を導入済みならそのままでOKです）。Largeモデルに変えたことで計算が重くなる分、学習エポック数40は据え置きでも1エポックあたり時間が延びる点に注意してください。必要に応じてエポック数を減らすか、学習時間を確保する計画を立てましょう。

4. 複数の設定ファイルがある場合の対応方針

プロジェクトにはBase+モデル用の設定ファイルが複数存在する可能性があります。代表的なのは以下の2つです：

微調整用フル設定: configs/sam2.1_training/sam2.1_hiera_b+_foodmix_finetune.yaml – 通常の学習用フル設定
GitHub
（Hydra経由で公式train.pyに渡すもの）。

メモリ最適化版設定: external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_foodmix_optimized.yaml – メモリリーク対策やパラメータ削減を施した設定
GitHub
。こちらは監視付きスクリプトで使用されます。

Largeモデルに切り替える際は、上記双方の設定ファイルをベースに新しいLarge用設定ファイルを作成することを推奨します。それぞれ以下のように対応してください。

通常微調整用のsam2.1_hiera_b+_foodmix_finetune.yamlをコピーして、sam2.1_hiera_l_foodmix_finetune.yamlという新ファイルを作成します（配置は元と同じくconfigs/sam2.1_training/ディレクトリ）。この中のmodel系項目をLarge用に書き換えます（先述のcheckpoint名・cfg名の変更、および必要に応じパラメータ微調整）。

メモリ最適化設定のsam2.1_hiera_b+_foodmix_optimized.yamlをコピーして、sam2.1_hiera_l_foodmix_optimized.yamlを作成します。こちらはSAM2.1公式リポジトリ内の設定パスに置く必要があります
GitHub
。つまり、external/sam2/sam2/configs/sam2.1_training/フォルダ内にファイルを配置します。同様にmodel項目をLarge用に変更し、さらにメモリ削減のためのパラメータ（前述のmax_num_objects等）も適宜見直した上で保存します。

以上2ファイルを用意することで、通常モード・メモリ最適化モードどちらでもLargeモデルを使った学習が行えるようになります。なお、sam2.1_hiera_b+_foodmix_simple.yamlのような簡易設定ファイルがある場合、これは主にデバッグ用途と思われますので、必要に応じて同様にコピーしてLarge版を作るか、Largeでは使用しないという選択でも構いません。中心となるのは上記のfinetune.yaml（完全版設定）とoptimized.yaml（メモリ対策版設定）です。

5. 修正後のファイル名例と配置先

上記の通り、新たに作成するLarge用設定ファイルの例としては**sam2.1_hiera_l_foodmix_finetune.yaml（通常学習用）およびsam2.1_hiera_l_foodmix_optimized.yaml**（メモリ最適化学習用）があります。

sam2.1_hiera_l_foodmix_finetune.yamlは、プロジェクト直下のconfigs/sam2.1_training/ディレクトリに配置します。名前はBase+版のファイル名からhiera_b+をhiera_lに変えればよいでしょう（他の部分はデータセット名や用途を表しているので同様で問題ありません）。

sam2.1_hiera_l_foodmix_optimized.yamlは、公式SAM2.1リポジトリ内のsam2/configs/sam2.1_training/フォルダに配置します
GitHub
。train_with_monitoring.sh経由でHydra設定を参照する際、このパスに置かれている必要があるためです。名前は上記と同様のルールで、..._hiera_b+_..._optimized.yamlを..._hiera_l_..._optimized.yamlに変更します。

新ファイルを配置したら、Git管理下であれば忘れずに追加しておきます。特にメモリ最適化版はexternal/sam2配下に置くため、プロジェクトルートから見るとサブモジュール領域になります。そこにユーザ定義の設定ファイルを追加した形になりますので、構成管理に注意してください（学習時にそのファイルが参照できれば問題ありません）。

6. トレーニングスクリプト／実行コマンドの変更点

Largeモデル用設定ファイルを用意した後は、学習実行スクリプトやコマンドライン引数で新しい設定ファイルを指すように変更する必要があります。現在の環境では、少なくとも以下の2通りの学習方法が想定されます。

通常学習スクリプト (scripts/20_train_food_sam2.sh) を使う場合: このスクリプト冒頭で、使用する設定ファイルパスが変数CONFIG_FILEとして指定されています
GitHub
（デフォルトではBase+用のsam2.1_hiera_b+_foodmix_finetune.yaml）。Largeモデルで学習する際には、この変数を新しいLarge用設定ファイルパス（例: configs/sam2.1_training/sam2.1_hiera_l_foodmix_finetune.yaml）に書き換えてからスクリプトを実行してください。また、スクリプト内では学習ログ出力先ディレクトリ名にモデル種別（foodmix_bplus_40ep 等）が含まれる場合があります
GitHub
。既存の訓練結果と混同しないよう、例えば**foodmix_l_40ep**のように名称を変える設定にするか、出力先を別ディレクトリに指定すると安全です（この部分はHydraのexperiment_log_dirや自動命名規則で決まるので、必要ならconfig内で変更）。

監視・メモリ最適化付きスクリプト (scripts/train_with_monitoring.sh) を使う場合: この場合、基本的にはコマンドライン引数で設定ファイルを指定できます。train_with_monitoring.shではデフォルトでCONFIG_NAME="sam2.1_training/sam2.1_hiera_b+_foodmix_optimized"がセットされています
GitHub
。Largeモデル用には、このデフォルトを一時的に書き換えるか、実行時に--config sam2.1_training/sam2.1_hiera_l_foodmix_optimizedオプションを付けて、新設定を指すようにします。例えばコマンド実行例として:

bash scripts/train_with_monitoring.sh --memory-optimized \
     --config sam2.1_training/sam2.1_hiera_l_foodmix_optimized \
     --auto-resume


のように指定できます（--memory-optimizedフラグでメモリ最適化版の学習を有効化
GitHub
GitHub
）。また、上記では--auto-resumeも付与していますが、これは途中で学習が中断した際に最新チェックポイントから自動再開するオプションです。Largeモデルでももちろん使用可能です。

なお、scripts/create_training_config.shはBase+用設定ファイルを自動生成するスクリプトですが
GitHub
GitHub
、Large用の設定生成には対応していません。このスクリプトをそのまま実行するとBase+用sam2.1_hiera_b+_foodmix_finetune.yamlが再生成されてしまうため、Largeモデル用設定は手動でコピー・編集する運用が確実です。どうしても自動化したい場合は、当該シェルスクリプト内のcheckpointファイル名とmodel_cfgパスをLarge用に書き換えて利用することもできますが、手動編集ミスが少ないことから上記のコピー&編集手順がおすすめです。

7. メモリ使用量・速度への影響と注意点

メモリ使用量: SAM2.1-Hiera-LargeはBase+に比べてモデル規模が大きいため、学習時のVRAM使用量が増加します。実際、本プロジェクトでもメモリ不足対策として解像度や同時物体数を削減することで最大75%のGPUメモリ削減効果を上げています
GitHub
。Largeモデルでは、例えば16GBのGPUではそのままではOOM（メモリ不足エラー）の可能性が高いです。そのため、前述したようにバッチサイズを極力小さくする、max_num_objectsを減らす、場合によっては画像サイズ（解像度）を下げる（例: 1024→640や512）といった対策が必要になるでしょう。A100 40GBクラスのGPUでも、Base+でバッチ4が可能だったところLargeではバッチ2程度に抑える必要があるかもしれません。学習実行中はこまめにGPU利用状況を監視し（本プロジェクトのtrain_with_monitoring.shがまさにそれを自動監視します）、使用率警告が出たら適宜設定を引き下げてください
GitHub
GitHub
。特にLargeモデルでは想定以上にメモリを消費する可能性があるため、余裕を持った設定でトライし、問題なければ徐々にバッチサイズ等を上げるアプローチが安全です。

 

速度への影響: Largeモデルは層が深くパラメータ数も多いため、1ステップあたりの計算時間がBase+より長くなります。おおよそ、モデル規模に比例して計算コストも増えるため、例えばBase+で1エポック数時間かかっていたなら、Largeではそれよりも長くなると見込まれます。学習全体の完了までに必要な時間が延びる点に注意し、学習計画を調整してください。時間短縮の工夫として、学習を途中で打ち切ってもよいようにチェックポイント保存間隔を短く設定する（例えばデフォルト500ステップを250ステップにする等
GitHub
）ことや、学習中間での評価頻度を下げてオーバーヘッドを減らすなどが考えられます。いずれにせよ、Largeモデル導入後は初回学習を短めのエポック数で試し、メモリと速度の挙動を確認することを強く推奨します。問題なく動作することを確認してから本番のエポック数で長時間の学習を実行すると、安全かつ効率的です。

8. 推論コード側の変更ポイント

学習だけでなく、推論・評価スクリプトにもLargeモデル対応が必要です。以下のポイントを確認してください。

可視化スクリプト (scripts/visualize_sam2_predictions.py): このスクリプトでは引数--checkpointと--model-cfgを指定して学習済みモデルの予測を可視化できます。デフォルトでは--model-cfg sam2_hiera_b+.yamlが使われ、Base+用設定を読むようになっています
GitHub
。Largeモデルを使う場合、コマンド実行時に--model-cfg sam2_hiera_l.yamlを指定してLarge用モデル設定を読むようにしてください。また、--checkpointもBase+用チェックポイントパス（例えば.../foodmix_bplus_40ep/checkpoint.pt）ではなく、Largeモデルでファインチューニングしたチェックポイント（例えば.../foodmix_l_40ep/checkpoint.ptなど）を指すよう変更します。これにより、可視化スクリプト内のbuild_sam2呼び出しにLargeモデルの構造と重みが渡され、正しい予測結果が得られます
GitHub
GitHub
。

評価スクリプト (scripts/21_eval_and_viz.py): このスクリプトでは引数--configでモデル設定ファイルを指定し、学習済みモデルの検証を行います。デフォルトでは--config external/sam2/configs/sam2.1/sam2.1_hiera_b+.yamlがハードコーディングされています
GitHub
。Largeモデルで評価する際は、--config external/sam2/configs/sam2.1/sam2.1_hiera_l.yamlを指定して実行してください（もしくはスクリプト内のデフォルト値を書き換えても構いません）。同様に、--checkpoint引数もBase+用ではなくLarge用に変更します
GitHub
。例えば、学習完了後に生成されたLargeモデルのチェックポイント（sam2_logs以下の該当ディレクトリに保存されています）をパスで指定します。

リアルタイム推論や他のスクリプト: もし他に独自の推論コードがある場合（例えばGUI経由でSAM2を使うケースなど）、モデル読み込み部分でBase+モデルの指定がハードコーディングされていないか確認してください。典型的にはbuild_sam2("sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt", ...)のような呼び出しになっている箇所です
GitHub
。その場合、設定ファイル名とチェックポイント名をLarge対応のものに変更する必要があります。同様に、SAM2ImagePredictorやSAM2AutomaticMaskGeneratorを使用するコードでも、内部で読み込むモデルが新しいLargeの重みに切り替わるように注意します（多くの場合は単に上記設定ファイルと重みパスを変えれば対応できます）。

最後に、推論時の注意点としてLargeモデルの方が計算コストが高いため、1画像あたりの推論時間が延びることを念頭に置いてください。必要なら、推論段階でも--device cuda（GPU使用）を指定しGPUで実行する、バッチ処理数を減らすなどの対策で時間増加に備えます。推論コード自体には大きな変更は不要ですが、使うモデルファイルの指定を漏れなくLarge版に切り替えることが重要です
GitHub
GitHub
。これらを適切に行えば、環境はSAM2.1-Hiera-Largeモデルを用いた安定かつ効率的なファインチューニングおよび推論に対応できるようになります。

 

参考資料: 本回答ではSoyaOda/sam2_1_food_finetuningリポジトリ内の設定ファイルとスクリプト、およびSAM2.1公式リポジトリの情報を参照しました
GitHub
GitHub
GitHub
GitHub
。上述の修正プランに沿って環境を構築すれば、SAM2.1-Hiera-Largeモデルを用いた食品画像セグメンテーションの微調整が円滑に行えるはずです。各ステップで不明点があれば、対応するソースコード内のコメントや本プロジェクトのREADME等も参照してください。