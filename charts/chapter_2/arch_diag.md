flowchart TD
    classDef source fill:whitesmoke,stroke:darkgray,stroke-width:2px,color:black,rx:5px,ry:5px
    classDef broker fill:floralwhite,stroke:tan,stroke-width:2px,color:black,rx:5px,ry:5px
    classDef online fill:aliceblue,stroke:dodgerblue,stroke-width:2px,color:black,rx:5px,ry:5px
    classDef offline fill:lightcyan,stroke:steelblue,stroke-width:2px,color:black,rx:5px,ry:5px
    classDef async fill:peachpuff,stroke:darkorange,stroke-width:2px,color:black,rx:5px,ry:5px
    classDef db fill:whitesmoke,stroke:slategray,stroke-width:2px,color:black,rx:5px,ry:5px

    Stream(["Strumień danych<br>tekstowych"]):::source
    Kafka{{"Apache Kafka<br>(broker i orkiestracja)"}}:::broker
    
    Stream -->|"Surowe teksty"| Kafka

    subgraph AsyncLayer ["Warstwa adaptacji (asynchroniczna)"]
        direction TB
        Buffer(["Zbuforowane okno<br>Danych"]):::async
        NSGA["Optymalizator NSGA-II<br>(algorytm genetyczny)"]:::async
        
        Buffer -.->|"2. Dane do ewaluacji"| NSGA
    end

    subgraph OnlineLayer ["Ścieżka bieżącego przetwarzania (online)"]
        direction TB
        Preproc["Moduł pre-processingu<br>(czyszczenie, SBERT, IPCA)"]:::online
        DenStream["Silnik klasteryzacji<br>Strumieniowej (DenStream)"]:::online
        DriftDet["Detektor dryfu pojęcia<br>(anomalie geometryczne)"]:::online
        
        Preproc -->|"Wektory Cech"| DenStream
        DenStream -->|"Metryki"| DriftDet
    end

    subgraph OfflineLayer ["Faza agregacji makroklastrów"]
        Agglom["Agregacja makroklastrów<br>(Grupowanie hierarhiczne)"]:::offline
    end

    Kafka -->|"Strumień dokumentów"| Preproc
    Kafka -.->|"Dane do okna optymalizacji"| Buffer

    DriftDet -->|"1. Zgłoszenie dryfu"| NSGA
    DenStream -.->|"&nbsp;&nbsp;Aktualny stan mikroklastrów"| Agglom
    
    NSGA -->|"3. Aktualizacja konfiguracji"| DenStream

    DB[("Warstwa persystencji<br>(PostgreSQL)")]:::db
    
    DriftDet -->|"Logowanie zdarzeń"| DB
    Agglom -->|"Archiwizacja makroklastrów"| DB
    NSGA -->|"Zapis najlepszych konfiguracji"| DB

    linkStyle default stroke:slategray,stroke-width:2px,color:black
