flowchart TD
    classDef source fill:whitesmoke,stroke:darkgray,stroke-width:2px,color:black
    classDef broker fill:floralwhite,stroke:tan,stroke-width:2px,color:black
    classDef online fill:aliceblue,stroke:dodgerblue,stroke-width:2px,color:black
    classDef offline fill:lightcyan,stroke:steelblue,stroke-width:2px,color:black
    classDef adapt fill:peachpuff,stroke:darkorange,stroke-width:2px,color:black
    classDef db fill:whitesmoke,stroke:slategray,stroke-width:2px,color:black

    Stream(["Strumień<br>tekstów"]):::source
    Kafka{{"Apache Kafka"}}:::broker

    subgraph Model ["Model"]
        direction TB
        Preproc["SBERT + IPCA"]:::online
        DenStream["DenStream<br>(faza online)"]:::online
        Agglom["Grupowanie<br>hierarchiczne<br>(faza offline)"]:::offline
        Preproc --> DenStream --> Agglom
    end

    Detector["Detektor<br>dryfu"]:::online
    Adapt["Adaptacja<br>bufor 500 dok.<br>IPCA + NSGA-II"]:::adapt
    DB[("PostgreSQL")]:::db

    Stream --> Kafka --> Preproc
    Agglom --> Detector
    Detector -. "dryf" .-> Adapt
    Adapt -. "nowy model" .-> Model
    Detector --> DB
    Adapt --> DB

    linkStyle default stroke:slategray,stroke-width:2px,color:black
